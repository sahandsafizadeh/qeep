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

			assertGradientEquals(t, act, exp)
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

			assertGradientEquals(t, act, exp)
		})

		t.Run("shared leaf fed through Add edges / BackPropagate / gradient accumulation not corrupted", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			s, err := a.Add(b)
			if err != nil {
				t.Fatal(err)
			}
			y, err := s.Add(a)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			acta := a.Gradient()
			actb := b.Gradient()

			expa, err := tensor.Full([]int{2, 2}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expb, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, acta, expa)
			assertGradientEquals(t, actb, expb)
		})

		t.Run("residual/skip connection: f(x) + x where x is a non-leaf / BackPropagate / gradient of x finalized before propagating to leaf", func(t *testing.T) {
			a, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			x := a.Scale(2.)
			f1 := x.Scale(3.)
			f2 := f1.Scale(4.)

			y, err := f2.Add(x)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := a.Gradient()

			exp, err := tensor.Full(nil, 26., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("element-wise value-dependent residual over a non-leaf tensor / BackPropagate / per-element gradient finalized before propagating", func(t *testing.T) {
			a, err := tensor.Of([][]float64{
				{1., 2.},
				{3., 4.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			x := a.Pow(2.)
			f1 := x.Scale(3.)
			f2 := f1.Scale(5.)

			y, err := f2.Add(x)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := a.Gradient()

			exp, err := tensor.Of([][]float64{
				{32., 64.},
				{96., 128.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("shared non-leaf with a long pending chain / BackPropagate / gradient not propagated until deep contributions arrive", func(t *testing.T) {
			a, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			c := a.Scale(2.)
			s1 := c.Scale(3.)
			s2 := s1.Scale(4.)
			s3 := s2.Scale(5.)

			y, err := s3.Add(c)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := a.Gradient()

			exp, err := tensor.Full(nil, 122., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("diamond DAG through a non-leaf intermediate / BackPropagate / gradient is finalized before propagating upstream (no double-counting)", func(t *testing.T) {
			a, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			m := a.Scale(2.)
			s := m.Sin()
			c := m.Cos()

			y, err := s.Add(c)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := a.Gradient()

			exp, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("value-dependent Mul diamond with shared non-leaf dequeued before its sibling contributes / BackPropagate / gradient correct", func(t *testing.T) {
			a, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			c := a.Scale(3.)
			e := c.Scale(5.)

			// c is the FIRST edge; a simple BFS would process c before the long path of e to c.
			y, err := c.Mul(e)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := a.Gradient()

			exp, err := tensor.Full(nil, 180., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("nested residuals: two stacked convergence points at different depths / BackPropagate / gradient accumulates correctly through both", func(t *testing.T) {
			a, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			h := a.Scale(2.)
			p1 := h.Scale(3.)
			p2 := p1.Scale(5.)

			s1, err := p2.Add(h)
			if err != nil {
				t.Fatal(err)
			}

			y, err := s1.Add(p1)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := a.Gradient()

			exp, err := tensor.Full(nil, 38., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== partial propagation ==============================

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

			assertGradientEquals(t, act, exp)

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

			assertGradientEquals(t, act, exp)
		})

		t.Run("multiple outputs sharing a non-leaf ancestor / BackPropagate on one then the other / non-leaf and leaf gradients accumulate across runs", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			m := x.Scale(2.)
			a := m.Scale(3.)
			b := m.Scale(5.)

			err = tensor.BackPropagate(a)
			if err != nil {
				t.Fatal(err)
			}

			actm := m.Gradient()
			actx := x.Gradient()

			expm, err := tensor.Full([]int{2, 2}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expx, err := tensor.Full([]int{2, 2}, 6., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, actm, expm)
			assertGradientEquals(t, actx, expx)

			err = tensor.BackPropagate(b)
			if err != nil {
				t.Fatal(err)
			}

			actm = m.Gradient()
			actx = x.Gradient()

			expm, err = tensor.Full([]int{2, 2}, 8., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expx, err = tensor.Full([]int{2, 2}, 16., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, actm, expm)
			assertGradientEquals(t, actx, expx)
		})

		t.Run("two independent chains from unrelated leaves / BackPropagate one then the other / unrelated leaf gradients remain isolated", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y1 := a.Scale(2.).Scale(3.)
			y2 := b.Scale(4.).Scale(5.)

			err = tensor.BackPropagate(y1)
			if err != nil {
				t.Fatal(err)
			}

			acta := a.Gradient()

			expa, err := tensor.Full([]int{2, 2}, 6., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, acta, expa)
			assertGradContext(t, b, true, false)

			err = tensor.BackPropagate(y2)
			if err != nil {
				t.Fatal(err)
			}

			actb := b.Gradient()
			acta = a.Gradient()

			expb, err := tensor.Full([]int{2, 2}, 20., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, actb, expb)
			assertGradientEquals(t, acta, expa)
		})

		t.Run("same output / BackPropagate twice without reset / gradient doubles", func(t *testing.T) {
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

			act := a.Gradient()

			exp, err := tensor.Full([]int{2, 2}, 6., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act = a.Gradient()

			exp, err = tensor.Full([]int{2, 2}, 12., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== tracking/resetting ==============================

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

			assertGradientEquals(t, act, exp)
		})

		t.Run("reset and rebuild a shared-leaf Add graph / BackPropagate, ResetGradContext, then BackPropagate again on a fresh identical graph / gradient matches a single fresh run", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			s1, err := a.Add(b)
			if err != nil {
				t.Fatal(err)
			}
			y1, err := s1.Add(a)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y1)
			if err != nil {
				t.Fatal(err)
			}

			a.ResetGradContext(true)
			b.ResetGradContext(true)

			s2, err := a.Add(b)
			if err != nil {
				t.Fatal(err)
			}
			y2, err := s2.Add(a)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y2)
			if err != nil {
				t.Fatal(err)
			}

			acta := a.Gradient()
			actb := b.Gradient()

			expa, err := tensor.Full([]int{2, 2}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expb, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, acta, expa)
			assertGradientEquals(t, actb, expb)
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
