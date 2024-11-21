package backward_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestChainGrad(t *testing.T) {
	runTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		a, err := tensor.Full([]int{3, 2}, 1., conf)
		if err != nil {
			t.Fatal(err)
		}

		y := a.Scale(2.).Scale(3.).Scale(5.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := a.Gradient()

		exp, err := tensor.Full([]int{3, 2}, 30., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestGradAccumulation(t *testing.T) {
	runTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		a, err := tensor.Full([]int{2, 3}, 1., conf)
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

		/* ------------------------------ */

		act := a.Gradient()

		exp, err := tensor.Full([]int{2, 3}, 10., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestUntrackedPaths(t *testing.T) {
	runTestLogicOnDevices(func(dev tensor.Device) {

		confU := &tensor.Config{
			Device:    dev,
			GradTrack: false,
		}

		confT := &tensor.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		a, err := tensor.Full([]int{2, 3}, 3., confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tensor.Full([]int{2, 3}, 2., confU)
		if err != nil {
			t.Fatal(err)
		}

		c, err := tensor.Full([]int{2, 3}, 1., confU)
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

		/* ------------------------------ */

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		if t1.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		if x1.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		if a.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* --------------- */

		if t2.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		if x3.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		if x2.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		if c.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		if b.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestValidationBackProp(t *testing.T) {
	runTestLogicOnDevices(func(_ tensor.Device) {

		/* ------------------------------ */

		err := tensor.BackPropagate(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "expected input tensor not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
