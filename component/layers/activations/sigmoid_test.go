package activations_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestSigmoid(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		activation := activations.NewSigmoid()

		/* ------------------------------ */

		x, err := tensor.Of(math.Inf(-1), conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := activation.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(-1e-10 < val && val < 1e-10) {
			t.Fatalf("expected scalar tensors value to be (0): got (%f)", val)
		}

		/* ------------------------------ */

		x, err = tensor.Of(math.Inf(+1), conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = activation.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(1.-1e-10 < val && val < 1.+1e-10) {
			t.Fatalf("expected scalar tensors value to be (1): got (%f)", val)
		}

		/* ------------------------------ */

		x, err = tensor.Of(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = activation.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		c := 1. / (1 + (1. / math.E))

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(c-1e-10 < val && val < c+1e-10) {
			t.Fatalf("expected scalar tensors value to be (1/(1+e^(-1))): got (%f)", val)
		}

		/* ------------------------------ */

		x, err = tensor.Zeros([]int{2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = activation.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.Full([]int{2, 3}, 0.5, conf)
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

func TestValidationSigmoid(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		activation := activations.NewSigmoid()

		/* ------------------------------ */

		x, err := tensor.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = activation.Forward()
		if err == nil {
			t.Fatalf("expected error because of not receiving one input tensor")
		} else if err.Error() != "Sigmoid input data validation failed: expected exactly one input tensor: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = activation.Forward(x, x)
		if err == nil {
			t.Fatalf("expected error because of not receiving one input tensor")
		} else if err.Error() != "Sigmoid input data validation failed: expected exactly one input tensor: got (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
