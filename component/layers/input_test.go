package layers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestInput(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Input with SeedFunc returning nil / Forward() / returns nil tensor", func(t *testing.T) {
			activation := layers.NewInput()
			activation.SeedFunc = func() tensor.Tensor { return nil }

			act, err := activation.Forward()
			if err != nil {
				t.Fatal(err)
			}

			if act != nil {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Input with SeedFunc returning [0,1,2,3,4,5] / Forward() / returns same values", func(t *testing.T) {
			activation := layers.NewInput()

			x, err := tensor.Of([]float64{0., 1., 2., 3., 4., 5.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			activation.SeedFunc = func() tensor.Tensor { return x }

			act, err := activation.Forward()
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{0., 1., 2., 3., 4., 5.}, &tensor.Config{Device: dev})
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

		t.Run("Input / Forward(x) with 1 input tensor / returns error: expected no input tensors", func(t *testing.T) {
			activation := layers.NewInput()
			activation.SeedFunc = func() tensor.Tensor { return nil }

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = activation.Forward(x)
			if err == nil {
				t.Fatal("expected error because of receiving input tensors")
			} else if err.Error() != "Input input data validation failed: expected no input tensors: got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Input / Forward(x, x) with 2 input tensors / returns error: expected no input tensors", func(t *testing.T) {
			activation := layers.NewInput()
			activation.SeedFunc = func() tensor.Tensor { return nil }

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = activation.Forward(x, x)
			if err == nil {
				t.Fatal("expected error because of receiving input tensors")
			} else if err.Error() != "Input input data validation failed: expected no input tensors: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
