package backward_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestBackPropagate(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("chain of operations / BackPropagate / gradient equals product of all scales", func(t *testing.T) {
			a, err := tensor.Full([]int{3, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := a.Scale(2.).Scale(3.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := a.Gradient()

			exp, err := tensor.Full([]int{3, 2}, 30., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("tensor used in multiple branches / BackPropagate / gradient accumulates from all branches", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			x1 := a.Scale(2.)
			x2 := a.Scale(3.)
			x3 := a.Scale(5.)

			y, err := tensor.Concat([]tensor.Tensor{x1, x2, x3}, 0)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := a.Gradient()

			exp, err := tensor.Full([]int{2, 3}, 10., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("diamond DAG: tracked tensor fans out to Sin and Cos then adds / BackPropagate / gradient equals cos(a) minus sin(a)", func(t *testing.T) {
			a, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			s := a.Sin()
			c := a.Cos()

			y, err := s.Add(c)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := a.Gradient()

			exp, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("multiple independent outputs from same tracked leaf / BackPropagate on one then the other / gradients accumulate without reset", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			a := x.Scale(2.)
			b := x.Scale(3.)

			err = tensor.BackPropagate(a)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full([]int{2, 2}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			err = tensor.BackPropagate(b)
			if err != nil {
				t.Fatal(err)
			}

			act = x.Gradient()

			exp, err = tensor.Full([]int{2, 2}, 5., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("mixed tracked/untracked computation graph / before BackPropagate / tracked paths have GradientTracked true, untracked false", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 3}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full([]int{2, 3}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			c, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			x1 := a.Scale(2.)
			x2 := b.Scale(3.)
			x3 := c.Scale(5.)

			t1, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}
			t2, err := x3.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Concat([]tensor.Tensor{t1, t2}, 0)
			if err != nil {
				t.Fatal(err)
			}

			assertGradContext(t, y, true, false)
			assertGradContext(t, t1, true, false)
			assertGradContext(t, x1, true, false)
			assertGradContext(t, a, true, false)
			assertGradContext(t, t2, false, false)
			assertGradContext(t, x3, false, false)
			assertGradContext(t, x2, false, false)
			assertGradContext(t, c, false, false)
			assertGradContext(t, b, false, false)
		})

		t.Run("mixed tracked/untracked computation graph / after BackPropagate / tracked nodes receive gradient, untracked remain nil", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 3}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full([]int{2, 3}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			c, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			x1 := a.Scale(2.)
			x2 := b.Scale(3.)
			x3 := c.Scale(5.)

			t1, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}
			t2, err := x3.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Concat([]tensor.Tensor{t1, t2}, 0)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			assertGradContext(t, y, true, true)
			assertGradContext(t, t1, true, true)
			assertGradContext(t, x1, true, true)
			assertGradContext(t, a, true, true)
			assertGradContext(t, t2, false, false)
			assertGradContext(t, x3, false, false)
			assertGradContext(t, x2, false, false)
			assertGradContext(t, c, false, false)
			assertGradContext(t, b, false, false)
		})

		t.Run("mixed tracked/untracked computation graph / after ResetGradContext / gradients cleared while tracking flags preserved", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 3}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full([]int{2, 3}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			c, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			x1 := a.Scale(2.)
			x2 := b.Scale(3.)
			x3 := c.Scale(5.)

			t1, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}
			t2, err := x3.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Concat([]tensor.Tensor{t1, t2}, 0)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			y.ResetGradContext(true)
			t1.ResetGradContext(true)
			x1.ResetGradContext(true)
			a.ResetGradContext(true)
			t2.ResetGradContext(false)
			x3.ResetGradContext(false)
			x2.ResetGradContext(false)
			c.ResetGradContext(false)
			b.ResetGradContext(false)

			assertGradContext(t, y, true, false)
			assertGradContext(t, t1, true, false)
			assertGradContext(t, x1, true, false)
			assertGradContext(t, a, true, false)
			assertGradContext(t, t2, false, false)
			assertGradContext(t, x3, false, false)
			assertGradContext(t, x2, false, false)
			assertGradContext(t, c, false, false)
			assertGradContext(t, b, false, false)
		})

		t.Run("re-run after ResetGradContext / BackPropagate twice on same leaf / second gradient equals first", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := a.Scale(2.).Scale(3.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			a.ResetGradContext(true)

			y2 := a.Scale(2.).Scale(3.)

			err = tensor.BackPropagate(y2)
			if err != nil {
				t.Fatal(err)
			}

			act := a.Gradient()

			exp, err := tensor.Full([]int{2, 2}, 6., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== validations ==============================

		t.Run("nil tensor / BackPropagate / returns error: unsupported tensor implementation", func(t *testing.T) {
			err := tensor.BackPropagate(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != "BackPropagate tensor implementation validation failed: unsupported tensor implementation" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

/* ----- helpers ----- */

func assertGradContext(t *testing.T, ten tensor.Tensor, tracked, gradSet bool) {
	t.Helper()

	if gradSet && ten.Gradient() == nil {
		t.Fatal("expected gradient not to be nil")
	} else if !gradSet && ten.Gradient() != nil {
		t.Fatal("expected gradient to be nil")
	}
	if tracked && !ten.GradientTracked() {
		t.Fatal("expected gradient to be tracked")
	} else if !tracked && ten.GradientTracked() {
		t.Fatal("expected gradient not to be tracked")
	}
}
