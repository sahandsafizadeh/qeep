package metrics_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/metrics"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestAccuracy(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		metric := metrics.NewAccuracy(&metrics.AccuracyConfig{OneHotMode: true})

		/* ------------------------------ */

		result, err := metric.Result()
		if err != nil {
			t.Fatal(err)
		}

		if !math.IsNaN(result) {
			t.Fatalf("expected result to be (NaN): got (%f)", result)
		}

		/* ------------------------------ */

		yp, err := tensor.TensorOf([][]float64{
			{1.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err := tensor.TensorOf([][]float64{
			{1.},
		}, conf)
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

		if !(1-1e-10 < result && result < 1+1e-10) {
			t.Fatalf("expected result to be (1): got (%f)", result)
		}

		/* ------------------------------ */

		yp, err = tensor.TensorOf([][]float64{
			{0.1, 0.9},
			{0.8, 0.2},
			{0.7, 0.3},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([][]float64{
			{0., 1.},
			{0., 1.},
			{0., 1.},
		}, conf)
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

		yp, err = tensor.TensorOf([][]float64{
			{0.2, 0.2, 0.2, 0.4},
			{0.6, 0.1, 0.1, 0.2},
			{0.1, 0.3, 0.1, 0.5},
			{0.2, 0.2, 0.4, 0.2},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([][]float64{
			{0., 0., 0., 1.},
			{0., 0., 1., 0.},
			{0., 1., 0., 0.},
			{1., 0., 0., 0.},
		}, conf)
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

		if !(0.375-1e-10 < result && result < 0.375+1e-10) {
			t.Fatalf("expected result to be (0.375): got (%f)", result)
		}

		/* ------------------------------ */

	})
}

func TestValidationAccuracy(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		metric := metrics.NewAccuracy(nil)

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

		err = metric.Accumulate(y1, y2)
		if err == nil {
			t.Fatalf("expected error because of tensors having more/less than two dimensions")
		} else if err.Error() != "Accuracy input data validation failed: expected input tensors to have exactly two dimensions (batch, class)" {
			t.Fatal("unexpected error message returned")
		}

		err = metric.Accumulate(y2, y5)
		if err == nil {
			t.Fatalf("expected error because of tensors having more/less than two dimensions")
		} else if err.Error() != "Accuracy input data validation failed: expected input tensors to have exactly two dimensions (batch, class)" {
			t.Fatal("unexpected error message returned")
		}

		err = metric.Accumulate(y2, y3)
		if err == nil {
			t.Fatalf("expected error because of tensors having unequal batch sizes")
		} else if err.Error() != "Accuracy input data validation failed: expected input tensor sizes to match along batch dimension: (1) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		err = metric.Accumulate(y2, y4)
		if err == nil {
			t.Fatalf("expected error because of tensors having unequal class sizes")
		} else if err.Error() != "Accuracy input data validation failed: expected input tensor sizes to match along class dimension: (1) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		err = metric.Accumulate(y4, y4)
		if err == nil {
			t.Fatalf("expected error because of tensors having class sizes unequal to (1) in not one-hot mode")
		} else if err.Error() != "Accuracy input data validation failed: expected input tensor sizes to be equal to (1) along class dimension when not in one-hot mode: got (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
