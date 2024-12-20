package activations_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestSoftmax(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		activation, err := activations.NewSoftmax(&activations.SoftmaxConfig{Dim: 0})
		if err != nil {
			t.Fatal(err)
		}

		x, err := tensor.Full([]int{4}, 0., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := activation.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.Full([]int{4}, 0.25, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		x, err = tensor.Full([]int{8, 6, 4}, 15., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = activation.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Full([]int{8, 6, 4}, 0.125, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		activation, err = activations.NewSoftmax(&activations.SoftmaxConfig{Dim: 1})
		if err != nil {
			t.Fatal(err)
		}

		x, err = tensor.Full([]int{6, 4}, 7., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = activation.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Full([]int{6, 4}, 0.25, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		x, err = tensor.Full([]int{4, 8, 6}, 1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = activation.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Full([]int{4, 8, 6}, 0.125, conf)
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

func TestValidationSoftmax(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		activation, err := activations.NewSoftmax(nil)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		x, err := tensor.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = activations.NewSoftmax(&activations.SoftmaxConfig{Dim: -1})
		if err == nil {
			t.Fatalf("expected error because of negative 'Dim'")
		} else if err.Error() != "Softmax config data validation failed: expected 'Dim' not to be negative: got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = activation.Forward()
		if err == nil {
			t.Fatalf("expected error because of not receiving one input tensor")
		} else if err.Error() != "Softmax input data validation failed: expected exactly one input tensor: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = activation.Forward(x, x)
		if err == nil {
			t.Fatalf("expected error because of not receiving one input tensor")
		} else if err.Error() != "Softmax input data validation failed: expected exactly one input tensor: got (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = activation.Forward(x)
		if err == nil {
			t.Fatalf("expected error because of input tensors shape not matching softmax 'Dim'")
		} else if err.Error() != "Softmax input data validation failed: expected input tensor shape to match 'Dim': [] !~ (0)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
