package metrics_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/metrics"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestMSE(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("MSE / Result() before any accumulation / result is NaN", func(t *testing.T) {
			metric := metrics.NewMSE()

			result := metric.Result()
			if !math.IsNaN(result) {
				t.Fatalf("expected result to be (NaN): got (%f)", result)
			}
		})

		t.Run("MSE / Accumulate([[0]] yp, [[1]] yt) / result near 1", func(t *testing.T) {
			metric := metrics.NewMSE()

			yp, err := tensor.Of([][]float64{
				{0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{
				{1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			result := metric.Result()
			if !(1.-1e-10 < result && result < 1.+1e-10) {
				t.Fatalf("expected result to be (1): got (%f)", result)
			}
		})

		t.Run("MSE / Accumulate([[0]] yp, [[1]] yt) then Accumulate([[2],[2],[0]] yp, [[2],[-1],[6]] yt) / result near 11.5", func(t *testing.T) {
			metric := metrics.NewMSE()

			yp, err := tensor.Of([][]float64{
				{0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{
				{1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			yp, err = tensor.Of([][]float64{
				{2.},
				{2.},
				{0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err = tensor.Of([][]float64{
				{2.},
				{-1.},
				{6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			result := metric.Result()
			if !(11.5-1e-10 < result && result < 11.5+1e-10) {
				t.Fatalf("expected result to be (11.5): got (%f)", result)
			}
		})

		t.Run("MSE / Accumulate([[0]] yp, [[1]] yt) then Accumulate([[2],[2],[0]] yp, [[2],[-1],[6]] yt) then Accumulate([[0,0,3],[1,0,3]] yp, [[0,-2,3],[1,0,-3]] yt) / result near 8.6", func(t *testing.T) {
			metric := metrics.NewMSE()

			yp, err := tensor.Of([][]float64{
				{0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{
				{1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			yp, err = tensor.Of([][]float64{
				{2.},
				{2.},
				{0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err = tensor.Of([][]float64{
				{2.},
				{-1.},
				{6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			yp, err = tensor.Of([][]float64{
				{0., 0., 3.},
				{1., 0., 3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err = tensor.Of([][]float64{
				{0., -2., 3.},
				{1., 0., -3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			result := metric.Result()
			if !(8.6-1e-10 < result && result < 8.6+1e-10) {
				t.Fatalf("expected result to be (8.6): got (%f)", result)
			}
		})

		// ============================== validations ==============================

		t.Run("MSE Accumulate with yp rank-1 and yt rank-2 / returns error: expected exactly two dimensions", func(t *testing.T) {
			metric := metrics.NewMSE()

			yp, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err == nil {
				t.Fatalf("expected error because of tensors having more/less than two dimensions")
			} else if err.Error() != "MSE input data validation failed: expected input tensors to have exactly two dimensions (batch, data)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("MSE Accumulate with yp rank-2 and yt rank-3 / returns error: expected exactly two dimensions", func(t *testing.T) {
			metric := metrics.NewMSE()

			yp, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Zeros([]int{1, 1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err == nil {
				t.Fatalf("expected error because of tensors having more/less than two dimensions")
			} else if err.Error() != "MSE input data validation failed: expected input tensors to have exactly two dimensions (batch, data)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("MSE Accumulate with unequal batch sizes / returns error: batch dimension mismatch", func(t *testing.T) {
			metric := metrics.NewMSE()

			yp, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Zeros([]int{2, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err == nil {
				t.Fatalf("expected error because of tensors having unequal batch sizes")
			} else if err.Error() != "MSE input data validation failed: expected input tensor sizes to match along batch dimension: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("MSE Accumulate with unequal data sizes / returns error: data dimension mismatch", func(t *testing.T) {
			metric := metrics.NewMSE()

			yp, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Zeros([]int{1, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err == nil {
				t.Fatalf("expected error because of tensors having unequal data sizes")
			} else if err.Error() != "MSE input data validation failed: expected input tensor sizes to match along data dimension: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
