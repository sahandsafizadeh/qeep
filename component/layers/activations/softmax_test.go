package activations_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestSoftmax(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("[4] uniform zeros input with Dim=0 / Forward() / output equals [4] tensor of 0.25", func(t *testing.T) {
			activation, err := activations.NewSoftmax(&activations.SoftmaxConfig{Dim: 0})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Full([]int{4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := activation.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{4}, 0.25, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[8,6,4] uniform input with Dim=0 / Forward() / output equals [8,6,4] tensor of 0.125", func(t *testing.T) {
			activation, err := activations.NewSoftmax(&activations.SoftmaxConfig{Dim: 0})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Full([]int{8, 6, 4}, 15., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := activation.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{8, 6, 4}, 0.125, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[6,4] uniform input with Dim=1 / Forward() / output equals [6,4] tensor of 0.25", func(t *testing.T) {
			activation, err := activations.NewSoftmax(&activations.SoftmaxConfig{Dim: 1})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Full([]int{6, 4}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := activation.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{6, 4}, 0.25, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[4,8,6] uniform input with Dim=1 / Forward() / output equals [4,8,6] tensor of 0.125", func(t *testing.T) {
			activation, err := activations.NewSoftmax(&activations.SoftmaxConfig{Dim: 1})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Full([]int{4, 8, 6}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := activation.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{4, 8, 6}, 0.125, &tensor.Config{Device: dev})
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

		t.Run("NewSoftmax(nil) / returns error: nil config", func(t *testing.T) {
			_, err := activations.NewSoftmax(nil)
			if err == nil {
				t.Fatal("expected error because of nil input config")
			} else if err.Error() != "Softmax config data validation failed: expected config not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewSoftmax with negative Dim / returns error: expected Dim not to be negative", func(t *testing.T) {
			_, err := activations.NewSoftmax(&activations.SoftmaxConfig{Dim: -1})
			if err == nil {
				t.Fatal("expected error because of negative 'Dim'")
			} else if err.Error() != "Softmax config data validation failed: expected 'Dim' not to be negative: got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("no input tensors / Forward() / returns error: expected exactly one input tensor", func(t *testing.T) {
			activation, err := activations.NewSoftmax(&activations.SoftmaxConfig{Dim: 0})
			if err != nil {
				t.Fatal(err)
			}

			_, err = activation.Forward()
			if err == nil {
				t.Fatal("expected error because of not receiving one input tensor")
			} else if err.Error() != "Softmax input data validation failed: expected exactly one input tensor: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("two input tensors / Forward() / returns error: expected exactly one input tensor", func(t *testing.T) {
			activation, err := activations.NewSoftmax(&activations.SoftmaxConfig{Dim: 0})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = activation.Forward(x, x)
			if err == nil {
				t.Fatal("expected error because of not receiving one input tensor")
			} else if err.Error() != "Softmax input data validation failed: expected exactly one input tensor: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar input with Dim=0 / Forward() / returns error: input shape does not match Dim", func(t *testing.T) {
			activation, err := activations.NewSoftmax(&activations.SoftmaxConfig{Dim: 0})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = activation.Forward(x)
			if err == nil {
				t.Fatal("expected error because of input tensors shape not matching softmax 'Dim'")
			} else if err.Error() != "Softmax input data validation failed: expected input tensor shape to match 'Dim': [] !~ (0)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
