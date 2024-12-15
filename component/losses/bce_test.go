package losses_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestBCE(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		loss := losses.NewBCE()

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

		yp, err = tensor.TensorOf([][]float64{{0.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([][]float64{{2.}}, conf)
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

		yp, err = tensor.TensorOf([][]float64{{1.5}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([][]float64{{-1.}}, conf)
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

		yp, err = tensor.TensorOf([][]float64{{0.1}, {0.2}, {0.8}, {0.9}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([][]float64{{0.}, {0.}, {1.}, {1.}}, conf)
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
		} else if !(0.163 < val && val < 0.165) {
			t.Fatalf("expected scalar tensors value to be (0.164): got (%f)", val)
		}

		/* ------------------------------ */

		yp, err = tensor.TensorOf([][]float64{{0.9}, {0.8}, {0.2}, {0.1}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([][]float64{{0.}, {0.}, {1.}, {1.}}, conf)
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
		} else if !(1.955 < val && val < 1.957) {
			t.Fatalf("expected scalar tensors value to be (1.956): got (%f)", val)
		}

		/* ------------------------------ */

	})
}

func TestValidationBCE(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		loss := losses.NewBCE()

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

		_, err = loss.Compute(y1, y2)
		if err == nil {
			t.Fatalf("expected error because of tensors having more/less than two dimensions")
		} else if err.Error() != "BCE input data validation failed: expected input tensors to have exactly two dimensions (batch, class=1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y2, y5)
		if err == nil {
			t.Fatalf("expected error because of tensors having more/less than two dimensions")
		} else if err.Error() != "BCE input data validation failed: expected input tensors to have exactly two dimensions (batch, class=1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y2, y3)
		if err == nil {
			t.Fatalf("expected error because of tensors having unequal batch sizes")
		} else if err.Error() != "BCE input data validation failed: expected input tensor sizes to match along batch dimension: (1) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y2, y4)
		if err == nil {
			t.Fatalf("expected error because of tensors having class sizes unequal to (1)")
		} else if err.Error() != "BCE input data validation failed: expected input tensor sizes to be equal to (1) along class dimension" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y4, y2)
		if err == nil {
			t.Fatalf("expected error because of tensors having class sizes unequal to (1)")
		} else if err.Error() != "BCE input data validation failed: expected input tensor sizes to be equal to (1) along class dimension" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
