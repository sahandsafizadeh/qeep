package layers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestInput(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		activation := layers.NewInput()

		/* ------------------------------ */

		activation.SeedFunc = func() tensor.Tensor { return nil }

		act, err := activation.Forward()
		if err != nil {
			t.Fatal(err)
		}

		if act != nil {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		x, err := tensor.TensorOf([]float64{0., 1., 2., 3., 4., 5.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		activation.SeedFunc = func() tensor.Tensor { return x }

		act, err = activation.Forward()
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.TensorOf([]float64{0., 1., 2., 3., 4., 5.}, conf)
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

func TestValidationInput(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		activation := layers.NewInput()
		activation.SeedFunc = func() tensor.Tensor { return nil }

		/* ------------------------------ */

		x, err := tensor.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = activation.Forward(x)
		if err == nil {
			t.Fatalf("expected error because of receiving input tensors")
		} else if err.Error() != "Input input data validation failed: expected no input tensors: got (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = activation.Forward(x, x)
		if err == nil {
			t.Fatalf("expected error because of receiving input tensors")
		} else if err.Error() != "Input input data validation failed: expected no input tensors: got (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
