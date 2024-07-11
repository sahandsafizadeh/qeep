package backward_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func TestChainGrad(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		a, err := tinit.Ones(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		y := a.Scale(2.).Scale(3.)

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := a.Gradient()

		exp, err := tinit.Full(conf, 6., 2)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		a, err := tinit.Full(conf, 1., 2)
		if err != nil {
			t.Fatal(err)
		}

		x1 := a.Scale(2.)
		x2 := a.Scale(3.)

		y, err := tinit.Concat([]tensor.Tensor{x1, x2}, 0)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := a.Gradient()

		exp, err := tinit.Full(conf, 5., 2)
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
