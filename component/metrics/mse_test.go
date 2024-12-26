package metrics_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/metrics"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestMSE(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		metric := metrics.NewMSE()

		/* ------------------------------ */

		result := metric.Result()

		if !(math.IsNaN(result)) {
			t.Fatalf("expected result to be (NaN): got (%f)", result)
		}

		/* ------------------------------ */

		yp, err := tensor.TensorOf([][]float64{
			{0.},
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

		result = metric.Result()

		if !(1.-1e-10 < result && result < 1.+1e-10) {
			t.Fatalf("expected result to be (1): got (%f)", result)
		}

		/* ------------------------------ */

		yp, err = tensor.TensorOf([][]float64{
			{2.},
			{2.},
			{0.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([][]float64{
			{2.},
			{-1.},
			{6.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		err = metric.Accumulate(yp, yt)
		if err != nil {
			t.Fatal(err)
		}

		result = metric.Result()

		if !(11.5-1e-10 < result && result < 11.5+1e-10) {
			t.Fatalf("expected result to be (11.5): got (%f)", result)
		}

		/* ------------------------------ */

		yp, err = tensor.TensorOf([][]float64{
			{0., 0., 3.},
			{1., 0., 3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		yt, err = tensor.TensorOf([][]float64{
			{0., -2., 3.},
			{1., 0., -3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		err = metric.Accumulate(yp, yt)
		if err != nil {
			t.Fatal(err)
		}

		result = metric.Result()

		if !(8.6-1e-10 < result && result < 8.6+1e-10) {
			t.Fatalf("expected result to be (8.6): got (%f)", result)
		}

		/* ------------------------------ */

	})
}

func TestValidationMSE(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		metric := metrics.NewMSE()

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
		} else if err.Error() != "MSE input data validation failed: expected input tensors to have exactly two dimensions (batch, data)" {
			t.Fatal("unexpected error message returned")
		}

		err = metric.Accumulate(y2, y5)
		if err == nil {
			t.Fatalf("expected error because of tensors having more/less than two dimensions")
		} else if err.Error() != "MSE input data validation failed: expected input tensors to have exactly two dimensions (batch, data)" {
			t.Fatal("unexpected error message returned")
		}

		err = metric.Accumulate(y2, y3)
		if err == nil {
			t.Fatalf("expected error because of tensors having unequal batch sizes")
		} else if err.Error() != "MSE input data validation failed: expected input tensor sizes to match along batch dimension: (1) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		err = metric.Accumulate(y2, y4)
		if err == nil {
			t.Fatalf("expected error because of tensors having unequal data sizes")
		} else if err.Error() != "MSE input data validation failed: expected input tensor sizes to match along data dimension: (1) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
