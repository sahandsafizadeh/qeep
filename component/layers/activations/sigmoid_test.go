package activations_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestSigmoid(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("scalar -Inf input / Forward() / output is near 0", func(t *testing.T) {
			activation := activations.NewSigmoid()

			x, err := tensor.Of(math.Inf(-1), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := activation.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of(-1e-10, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of(1e-10, &tensor.Config{Device: dev})
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

		t.Run("scalar +Inf input / Forward() / output is near 1", func(t *testing.T) {
			activation := activations.NewSigmoid()

			x, err := tensor.Of(math.Inf(+1), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := activation.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of(1.-1e-10, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of(1.+1e-10, &tensor.Config{Device: dev})
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

		t.Run("scalar 1 input / Forward() / output is near 1/(1+e^(-1))", func(t *testing.T) {
			activation := activations.NewSigmoid()

			x, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := activation.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			c := 1. / (1 + (1. / math.E))

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

		t.Run("[2,3] zeros input / Forward() / output equals [2,3] tensor of 0.5", func(t *testing.T) {
			activation := activations.NewSigmoid()

			x, err := tensor.Zeros([]int{2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := activation.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{2, 3}, 0.5, &tensor.Config{Device: dev})
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

		t.Run("no input tensors / Forward() / returns error: expected exactly one input tensor", func(t *testing.T) {
			activation := activations.NewSigmoid()

			_, err := activation.Forward()
			if err == nil {
				t.Fatalf("expected error because of not receiving one input tensor")
			} else if err.Error() != "Sigmoid input data validation failed: expected exactly one input tensor: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("two input tensors / Forward() / returns error: expected exactly one input tensor", func(t *testing.T) {
			activation := activations.NewSigmoid()

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = activation.Forward(x, x)
			if err == nil {
				t.Fatalf("expected error because of not receiving one input tensor")
			} else if err.Error() != "Sigmoid input data validation failed: expected exactly one input tensor: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
