package activations_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestTanh(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		activation := activations.NewTanh()

		/* ------------------------------ */

		x, err := tensor.Of(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := activation.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		c := (math.E*math.E - 1) / (math.E*math.E + 1)

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(c-1e-10 < val && val < c+1e-10) {
			t.Fatalf("expected scalar tensors value to be ((e^2-1)/(e^2+1)): got (%f)", val)
		}

		/* ------------------------------ */

	})
}

func TestValidationTanh(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		activation := activations.NewTanh()

		/* ------------------------------ */

		x, err := tensor.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = activation.Forward()
		if err == nil {
			t.Fatalf("expected error because of not receiving one input tensor")
		} else if err.Error() != "Tanh input data validation failed: expected exactly one input tensor: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = activation.Forward(x, x)
		if err == nil {
			t.Fatalf("expected error because of not receiving one input tensor")
		} else if err.Error() != "Tanh input data validation failed: expected exactly one input tensor: got (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
