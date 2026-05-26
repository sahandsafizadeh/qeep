package activations_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestLeakyRelu(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("LeakyRelu(M=0.5) on mixed negative/zero/positive matrix / Forward() / scales negatives by 0.5, passes positives unchanged", func(t *testing.T) {
			activation := activations.NewLeakyRelu(&activations.LeakyReluConfig{M: 0.5})

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
				{-1., -0.5, -0.005},
				{0., 0., 0.},
				{0.01, 1., 2.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		// ============================== validations ==============================

		t.Run("LeakyRelu(nil) / Forward() with no input tensors / returns error: expected exactly one input tensor", func(t *testing.T) {
			activation := activations.NewLeakyRelu(nil)

			_, err := activation.Forward()
			if err == nil {
				t.Fatalf("expected error because of not receiving one input tensor")
			} else if err.Error() != "LeakyRelu input data validation failed: expected exactly one input tensor: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("LeakyRelu(nil) / Forward() with two input tensors / returns error: expected exactly one input tensor", func(t *testing.T) {
			activation := activations.NewLeakyRelu(nil)

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = activation.Forward(x, x)
			if err == nil {
				t.Fatalf("expected error because of not receiving one input tensor")
			} else if err.Error() != "LeakyRelu input data validation failed: expected exactly one input tensor: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
