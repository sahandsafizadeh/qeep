package metrics_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/metrics"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestAccuracy(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Accuracy (non-one-hot) / Result() before any accumulation / result is NaN", func(t *testing.T) {
			metric := metrics.NewAccuracy(&metrics.AccuracyConfig{OneHotMode: false})

			result := metric.Result()
			if !math.IsNaN(result) {
				t.Fatalf("expected result to be (NaN): got (%f)", result)
			}
		})

		t.Run("Accuracy (non-one-hot) / Accumulate([[0]] yp, [[0]] yt) / result near 1", func(t *testing.T) {
			metric := metrics.NewAccuracy(&metrics.AccuracyConfig{OneHotMode: false})

			yp, err := tensor.Of([][]float64{{0.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{{0.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			result := metric.Result()
			if !(1-1e-10 < result && result < 1+1e-10) {
				t.Fatalf("expected result to be (1): got (%f)", result)
			}
		})

		t.Run("Accuracy (non-one-hot) / Accumulate([[0]] yp, [[0]] yt) then Accumulate([[0.1],[0.8],[0.5],[0.4999]] yp, [[0],[0],[0],[0]] yt) / result near 0.6", func(t *testing.T) {
			metric := metrics.NewAccuracy(&metrics.AccuracyConfig{OneHotMode: false})

			yp, err := tensor.Of([][]float64{{0.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{{0.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			yp, err = tensor.Of([][]float64{{0.1}, {0.8}, {0.5}, {0.4999}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err = tensor.Of([][]float64{{0.}, {0.}, {0.}, {0.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			result := metric.Result()
			if !(0.6-1e-10 < result && result < 0.6+1e-10) {
				t.Fatalf("expected result to be (0.6): got (%f)", result)
			}
		})

		t.Run("Accuracy (one-hot) / Result() before any accumulation / result is NaN", func(t *testing.T) {
			metric := metrics.NewAccuracy(&metrics.AccuracyConfig{OneHotMode: true})

			result := metric.Result()

			if !math.IsNaN(result) {
				t.Fatalf("expected result to be (NaN): got (%f)", result)
			}
		})

		t.Run("Accuracy (one-hot) / Accumulate([[1]] yp, [[1]] yt) / result near 1", func(t *testing.T) {
			metric := metrics.NewAccuracy(&metrics.AccuracyConfig{OneHotMode: true})

			yp, err := tensor.Of([][]float64{
				{1.},
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

			if !(1-1e-10 < result && result < 1+1e-10) {
				t.Fatalf("expected result to be (1): got (%f)", result)
			}
		})

		t.Run("Accuracy (one-hot) / Accumulate([[1]] yp, [[1]] yt) then Accumulate([[0.1,0.9],[0.8,0.2],[0.7,0.3]] yp, [[0,1],[0,1],[0,1]] yt) / result near 0.5", func(t *testing.T) {
			metric := metrics.NewAccuracy(&metrics.AccuracyConfig{OneHotMode: true})

			yp, err := tensor.Of([][]float64{
				{1.},
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
				{0.1, 0.9},
				{0.8, 0.2},
				{0.7, 0.3},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err = tensor.Of([][]float64{
				{0., 1.},
				{0., 1.},
				{0., 1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			result := metric.Result()
			if !(0.5-1e-10 < result && result < 0.5+1e-10) {
				t.Fatalf("expected result to be (0.5): got (%f)", result)
			}
		})

		t.Run("Accuracy (one-hot) / Accumulate([[1]] yp, [[1]] yt) then Accumulate 3x2 then Accumulate 4x4 / result near 0.375", func(t *testing.T) {
			metric := metrics.NewAccuracy(&metrics.AccuracyConfig{OneHotMode: true})

			yp, err := tensor.Of([][]float64{
				{1.},
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
				{0.1, 0.9},
				{0.8, 0.2},
				{0.7, 0.3},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err = tensor.Of([][]float64{
				{0., 1.},
				{0., 1.},
				{0., 1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			yp, err = tensor.Of([][]float64{
				{0.2, 0.2, 0.2, 0.4},
				{0.6, 0.1, 0.1, 0.2},
				{0.1, 0.3, 0.1, 0.5},
				{0.2, 0.2, 0.4, 0.2},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err = tensor.Of([][]float64{
				{0., 0., 0., 1.},
				{0., 0., 1., 0.},
				{0., 1., 0., 0.},
				{1., 0., 0., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			result := metric.Result()
			if !(0.375-1e-10 < result && result < 0.375+1e-10) {
				t.Fatalf("expected result to be (0.375): got (%f)", result)
			}
		})

		// ============================== validations ==============================

		t.Run("Accuracy Accumulate with yp rank-1 and yt rank-2 / returns error: expected exactly two dimensions", func(t *testing.T) {
			metric := metrics.NewAccuracy(nil)

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
			} else if err.Error() != "Accuracy input data validation failed: expected input tensors to have exactly two dimensions (batch, class)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Accuracy Accumulate with yp rank-2 and yt rank-3 / returns error: expected exactly two dimensions", func(t *testing.T) {
			metric := metrics.NewAccuracy(nil)

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
			} else if err.Error() != "Accuracy input data validation failed: expected input tensors to have exactly two dimensions (batch, class)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Accuracy Accumulate with unequal batch sizes / returns error: batch dimension mismatch", func(t *testing.T) {
			metric := metrics.NewAccuracy(nil)

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
			} else if err.Error() != "Accuracy input data validation failed: expected input tensor sizes to match along batch dimension: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Accuracy Accumulate with unequal class sizes / returns error: class dimension mismatch", func(t *testing.T) {
			metric := metrics.NewAccuracy(nil)

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
				t.Fatalf("expected error because of tensors having unequal class sizes")
			} else if err.Error() != "Accuracy input data validation failed: expected input tensor sizes to match along class dimension: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Accuracy Accumulate with class size 2 in non-one-hot mode / returns error: class size not equal to 1 in non-one-hot mode", func(t *testing.T) {
			metric := metrics.NewAccuracy(nil)

			yp, err := tensor.Zeros([]int{1, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Zeros([]int{1, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = metric.Accumulate(yp, yt)
			if err == nil {
				t.Fatalf("expected error because of tensors having class sizes unequal to (1) in not one-hot mode")
			} else if err.Error() != "Accuracy input data validation failed: expected input tensor sizes to be equal to (1) along class dimension when not in one-hot mode: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
