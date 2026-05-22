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
				t.Fatalf("expected tensors to be equal")
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
				t.Fatalf("expected tensors to be equal")
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

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}

			if t1.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if !t1.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}

			if x1.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if !x1.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}

			if a.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if !a.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}

			if t2.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if t2.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}

			if x3.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if x3.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}

			if x2.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if x2.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}

			if c.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if c.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}

			if b.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if b.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
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

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			} else if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}

			if t1.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			} else if !t1.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}

			if x1.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			} else if !x1.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}

			if a.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			} else if !a.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}

			if t2.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if t2.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}

			if x3.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if x3.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}

			if x2.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if x2.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}

			if c.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if c.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}

			if b.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if b.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
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

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}

			if t1.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if !t1.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}

			if x1.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if !x1.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}

			if a.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if !a.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}

			if t2.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if t2.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}

			if x3.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if x3.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}

			if x2.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if x2.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}

			if c.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if c.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}

			if b.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			} else if b.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("nil tensor / BackPropagate / returns error: unsupported tensor implementation", func(t *testing.T) {
			err := tensor.BackPropagate(nil)
			if err == nil {
				t.Fatalf("expected error because of nil input tensor")
			} else if err.Error() != "BackPropagate tensor implementation validation failed: unsupported tensor implementation" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
