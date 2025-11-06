package layers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestDropout(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		confU := &tensor.Config{
			Device:    dev,
			GradTrack: false,
		}

		confT := &tensor.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		layer, err := layers.NewDropout(nil)
		if err != nil {
			t.Fatal(err)
		}

		x, err := tensor.Of([][]float64{{3., 1., 2.}}, confU)
		if err != nil {
			t.Fatal(err)
		}

		act, err := layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.Of([][]float64{{3., 1., 2.}}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		layer, err = layers.NewDropout(nil)
		if err != nil {
			t.Fatal(err)
		}

		x, err = tensor.Of(-5., confT)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		exp1, err := tensor.Of(0., confT)
		if err != nil {
			t.Fatal(err)
		}

		exp2, err := tensor.Of(-10., confT)
		if err != nil {
			t.Fatal(err)
		}

		eq1, err := act.Equals(exp1)
		if err != nil {
			t.Fatal(err)
		}

		eq2, err := act.Equals(exp2)
		if err != nil {
			t.Fatal(err)
		}

		if !eq1 && !eq2 {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		layer, err = layers.NewDropout(&layers.DropoutConfig{
			Rate: 0.9,
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err = tensor.Of([][]float64{{1.}}, confT)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		if val, err := act.At(0, 0); err != nil {
			t.Fatal(err)
		} else if !((-0.1 < val && val < 0.1) || (9.9 < val && val < 10.1)) {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		layer, err = layers.NewDropout(&layers.DropoutConfig{
			Rate: 0,
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err = tensor.Full([]int{10, 10, 100}, 3., confT)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Full([]int{10, 10, 100}, 3., confT)
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

func TestValidationDropout(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		_, err := layers.NewDropout(&layers.DropoutConfig{
			Rate: -0.1,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'Rate'")
		} else if err.Error() != "Dropout config data validation failed: expected 'Rate' to be in range [0,1): got (-0.100000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layers.NewDropout(&layers.DropoutConfig{
			Rate: 1,
		})
		if err == nil {
			t.Fatalf("expected error because of 'Rate' being greater than or equal to (1)")
		} else if err.Error() != "Dropout config data validation failed: expected 'Rate' to be in range [0,1): got (1.000000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layers.NewDropout(&layers.DropoutConfig{
			Rate: 1.001,
		})
		if err == nil {
			t.Fatalf("expected error because of 'Rate' being greater than or equal to (1)")
		} else if err.Error() != "Dropout config data validation failed: expected 'Rate' to be in range [0,1): got (1.001000)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		layer, err := layers.NewDropout(nil)
		if err != nil {
			t.Fatal(err)
		}

		x, err := tensor.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = layer.Forward()
		if err == nil {
			t.Fatalf("expected error because of not receiving one input tensor")
		} else if err.Error() != "Dropout input data validation failed: expected exactly one input tensor: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layer.Forward(x, x)
		if err == nil {
			t.Fatalf("expected error because of not receiving one input tensor")
		} else if err.Error() != "Dropout input data validation failed: expected exactly one input tensor: got (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
