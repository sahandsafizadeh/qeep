package activations_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestTanh(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("scalar 1 input / Forward() / output is near (e^2-1)/(e^2+1)", func(t *testing.T) {
			activation := activations.NewTanh()

			x, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := activation.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			c := (math.E*math.E - 1) / (math.E*math.E + 1)

			expl, err := tensor.Of(c-1e-10, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of(c+1e-10, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatalf("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatalf("expected output to be in range")
			}
		})

		// ============================== validations ==============================

		t.Run("no input tensors / Forward() / returns error: expected exactly one input tensor", func(t *testing.T) {
			activation := activations.NewTanh()

			_, err := activation.Forward()
			if err == nil {
				t.Fatalf("expected error because of not receiving one input tensor")
			} else if err.Error() != "Tanh input data validation failed: expected exactly one input tensor: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("two input tensors / Forward() / returns error: expected exactly one input tensor", func(t *testing.T) {
			activation := activations.NewTanh()

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = activation.Forward(x, x)
			if err == nil {
				t.Fatalf("expected error because of not receiving one input tensor")
			} else if err.Error() != "Tanh input data validation failed: expected exactly one input tensor: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
