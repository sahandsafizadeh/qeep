package losses_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestMSE(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		loss := losses.NewMSE()

		/* ------------------------------ */

		yp, err := tensor.TensorOf([][]float64{{0.5}}, conf)
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

		exp, err := tensor.TensorOf(0.25, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		yp, err = tensor.TensorOf([][]float64{{2.}, {2.}, {0.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([][]float64{{2.}, {-1.}, {6.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = loss.Compute(yp, yt)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.TensorOf(15., conf)
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

func TestValidationMSE(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		loss := losses.NewMSE()

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
		} else if err.Error() != "MSE input data validation failed: expected input tensors not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y1, nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "MSE input data validation failed: expected input tensors not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y1, y2)
		if err == nil {
			t.Fatalf("expected error because of tensors having more/less than two dimensions")
		} else if err.Error() != "MSE input data validation failed: expected input tensors to have exactly two dimensions (batch, class)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y2, y5)
		if err == nil {
			t.Fatalf("expected error because of tensors having more/less than two dimensions")
		} else if err.Error() != "MSE input data validation failed: expected input tensors to have exactly two dimensions (batch, class)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y2, y3)
		if err == nil {
			t.Fatalf("expected error because of tensors having unequal batch sizes")
		} else if err.Error() != "MSE input data validation failed: expected input tensor sizes to match along batch dimension: (1) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y2, y4)
		if err == nil {
			t.Fatalf("expected error because of tensors having class sizes unequal to (1)")
		} else if err.Error() != "MSE input data validation failed: expected input tensor sizes to be equal to (1) along class dimension" {
			t.Fatal("unexpected error message returned")
		}

		_, err = loss.Compute(y4, y2)
		if err == nil {
			t.Fatalf("expected error because of tensors having class sizes unequal to (1)")
		} else if err.Error() != "MSE input data validation failed: expected input tensor sizes to be equal to (1) along class dimension" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
