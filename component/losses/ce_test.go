package losses_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestCE(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		loss := losses.NewCE()

		/* ------------------------------ */

		yp, err := tensor.TensorOf([][]float64{{0.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err := tensor.TensorOf([][]float64{{0.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := loss.Compute(yp, yt)
		if err != nil {
			t.Fatal(err)
		}

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.-1e-10 < val && val < 0.+1e-10) {
			t.Fatalf("expected scalar tensors value to be (0): got (%f)", val)
		}

		/* ------------------------------ */

		yp, err = tensor.TensorOf([][]float64{{1.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([][]float64{{1.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = loss.Compute(yp, yt)
		if err != nil {
			t.Fatal(err)
		}

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.-1e-10 < val && val < 0.+1e-10) {
			t.Fatalf("expected scalar tensors value to be (0): got (%f)", val)
		}

		/* ------------------------------ */

		yp, err = tensor.TensorOf([][]float64{{0., 1.5}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([][]float64{{2., -1.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = loss.Compute(yp, yt)
		if err != nil {
			t.Fatal(err)
		}

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(27.630 < val && val < 27.632) {
			t.Fatalf("expected scalar tensors value to be (27.632): got (%f)", val)
		}

		/* ------------------------------ */

		yp, err = tensor.TensorOf([][]float64{
			{0.1, 0.8, 0.1},
			{0.1, 0.2, 0.7},
			{0.9, 0.1, 0.0},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([][]float64{
			{0., 1., 0.},
			{0., 0., 1.},
			{1., 0., 0.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = loss.Compute(yp, yt)
		if err != nil {
			t.Fatal(err)
		}

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.227 < val && val < 0.229) {
			t.Fatalf("expected scalar tensors value to be (0.228): got (%f)", val)
		}

		/* ------------------------------ */

		yp, err = tensor.TensorOf([][]float64{
			{0.4, 0.1, 0.5},
			{0.4, 0.4, 0.2},
			{0.3, 0.3, 0.4},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([][]float64{
			{0., 1., 0.},
			{0., 0., 1.},
			{1., 0., 0.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = loss.Compute(yp, yt)
		if err != nil {
			t.Fatal(err)
		}

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(1.706 < val && val < 1.707) {
			t.Fatalf("expected scalar tensors value to be (1.705): got (%f)", val)
		}

		/* ------------------------------ */

	})
}

func TestValidationCE(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		loss := losses.NewCE()

		/* ------------------------------ */

		y1, err := tensor.Zeros([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		y2, err := tensor.Zeros([]int{1, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		y3, err := tensor.Zeros([]int{2, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		y4, err := tensor.Zeros([]int{1, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		y5, err := tensor.Zeros([]int{1, 1, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = loss.Compute(nil, y1)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "CE input data validation failed: expected input tensors not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y1, nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "CE input data validation failed: expected input tensors not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y1, y2)
		if err == nil {
			t.Fatalf("expected error because of tensors having more/less than two dimensions")
		} else if err.Error() != "CE input data validation failed: expected input tensors to have exactly two dimensions (batch, class)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y2, y5)
		if err == nil {
			t.Fatalf("expected error because of tensors having more/less than two dimensions")
		} else if err.Error() != "CE input data validation failed: expected input tensors to have exactly two dimensions (batch, class)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y2, y3)
		if err == nil {
			t.Fatalf("expected error because of tensors having unequal batch sizes")
		} else if err.Error() != "CE input data validation failed: expected input tensor sizes to match along batch dimension: (1) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y2, y4)
		if err == nil {
			t.Fatalf("expected error because of tensors having unequal class sizes")
		} else if err.Error() != "CE input data validation failed: expected input tensor sizes to match along class dimension: (1) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
