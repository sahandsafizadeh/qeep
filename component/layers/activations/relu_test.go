package activations_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestRelu(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("[3,3] mixed input / Forward() / negatives and zero become 0, positives unchanged", func(t *testing.T) {
			activation := activations.NewRelu()

			x, err := tensor.Of([][]float64{
				{-2., -1., -0.01},
				{0., 0., 0.},
				{0.01, 1., 2.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := activation.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{0., 0., 0.},
				{0., 0., 0.},
				{0.01, 1., 2.},
			}, &tensor.Config{Device: dev})
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

		t.Run("no input tensors / Forward() / returns error: expected exactly one input tensor", func(t *testing.T) {
			activation := activations.NewRelu()

			_, err := activation.Forward()
			if err == nil {
				t.Fatal("expected error because of not receiving one input tensor")
			} else if err.Error() != "Relu input data validation failed: expected exactly one input tensor: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("two input tensors / Forward() / returns error: expected exactly one input tensor", func(t *testing.T) {
			activation := activations.NewRelu()

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = activation.Forward(x, x)
			if err == nil {
				t.Fatal("expected error because of not receiving one input tensor")
			} else if err.Error() != "Relu input data validation failed: expected exactly one input tensor: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
