package metrics_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/metrics"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestAccuracy(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		metric := metrics.NewAccuracy()

		/* ------------------------------ */

		result, err := metric.Result()
		if err != nil {
			t.Fatal(err)
		}

		if !(-1e-10 < result && result < 1e-10) {
			t.Fatalf("expected result to be (0): got (%f)", result)
		}

		/* ------------------------------ */

		yp, err := tensor.TensorOf([]float64{0.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err := tensor.TensorOf([]float64{1.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		err = metric.Accumulate(yp, yt)
		if err != nil {
			t.Fatal(err)
		}

		result, err = metric.Result()
		if err != nil {
			t.Fatal(err)
		}

		if !(-1e-10 < result && result < 1e-10) {
			t.Fatalf("expected result to be (0): got (%f)", result)
		}

		/* ------------------------------ */

		yp, err = tensor.TensorOf([]float64{0., 1., 2.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([]float64{1., 1., 2.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		err = metric.Accumulate(yp, yt)
		if err != nil {
			t.Fatal(err)
		}

		result, err = metric.Result()
		if err != nil {
			t.Fatal(err)
		}

		if !(0.5-1e-10 < result && result < 0.5+1e-10) {
			t.Fatalf("expected result to be (0.5): got (%f)", result)
		}

		/* ------------------------------ */

		yp, err = tensor.TensorOf([]float64{3., 2., 1., 0.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([]float64{3., 2., 1., 0.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		err = metric.Accumulate(yp, yt)
		if err != nil {
			t.Fatal(err)
		}

		result, err = metric.Result()
		if err != nil {
			t.Fatal(err)
		}

		if !(0.75-1e-10 < result && result < 0.75+1e-10) {
			t.Fatalf("expected result to be (0.75): got (%f)", result)
		}

		/* ------------------------------ */

	})
}

func TestValidationAccuracy(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		metric := metrics.NewAccuracy()

		/* ------------------------------ */

		y1, err := tensor.Zeros([]int{}, conf)
		if err != nil {
			t.Fatal(err)
		}

		y2, err := tensor.Zeros([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		y3, err := tensor.Zeros([]int{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		y4, err := tensor.Zeros([]int{1, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		err = metric.Accumulate(nil, y1)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "Accuracy input data validation failed: expected input tensors not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		err = metric.Accumulate(y1, nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != "Accuracy input data validation failed: expected input tensors not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		err = metric.Accumulate(y1, y2)
		if err == nil {
			t.Fatalf("expected error because of tensors having more/less than one dimension")
		} else if err.Error() != "Accuracy input data validation failed: expected input tensors to have exactly one dimension (batch)" {
			t.Fatal("unexpected error message returned")
		}

		err = metric.Accumulate(y2, y4)
		if err == nil {
			t.Fatalf("expected error because of tensors having more/less than one dimension")
		} else if err.Error() != "Accuracy input data validation failed: expected input tensors to have exactly one dimension (batch)" {
			t.Fatal("unexpected error message returned")
		}

		err = metric.Accumulate(y2, y3)
		if err == nil {
			t.Fatalf("expected error because of tensors having unequal batch sizes")
		} else if err.Error() != "Accuracy input data validation failed: expected input tensor sizes to match along batch dimension: (1) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
