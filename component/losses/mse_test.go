package losses_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestMSE(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("MSE / Compute([[0.5]] yp, [[0]] yt) / output equals 0.25", func(t *testing.T) {
			loss := losses.NewMSE()

			yp, err := tensor.Of([][]float64{
				{0.5},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{
				{0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := loss.Compute(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(0.25, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("MSE / Compute([[2],[2],[0]] yp, [[2],[-1],[6]] yt) / output equals 15", func(t *testing.T) {
			loss := losses.NewMSE()

			yp, err := tensor.Of([][]float64{
				{2.},
				{2.},
				{0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{
				{2.},
				{-1.},
				{6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := loss.Compute(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(15., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("MSE / Compute(4x4 yp, 4x4 yt with partial overlap) / output equals 2.5", func(t *testing.T) {
			loss := losses.NewMSE()

			yp, err := tensor.Of([][]float64{
				{0., 1., 2., 3.},
				{4., 5., 6., 7.},
				{4., 5., 6., 7.},
				{3., 2., 1., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{
				{3., 2., 1., 0.},
				{4., 5., 6., 7.},
				{7., 6., 5., 4.},
				{3., 2., 1., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := loss.Compute(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(2.5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== validations ==============================

		t.Run("MSE Compute with yp rank-1 and yt rank-2 / returns error: expected exactly two dimensions", func(t *testing.T) {
			loss := losses.NewMSE()

			yp, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = loss.Compute(yp, yt)
			if err == nil {
				t.Fatal("expected error because of tensors having more/less than two dimensions")
			} else if err.Error() != "MSE input data validation failed: expected input tensors to have exactly two dimensions (batch, data)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("MSE Compute with yp rank-2 and yt rank-3 / returns error: expected exactly two dimensions", func(t *testing.T) {
			loss := losses.NewMSE()

			yp, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Zeros([]int{1, 1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = loss.Compute(yp, yt)
			if err == nil {
				t.Fatal("expected error because of tensors having more/less than two dimensions")
			} else if err.Error() != "MSE input data validation failed: expected input tensors to have exactly two dimensions (batch, data)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("MSE Compute with unequal batch sizes / returns error: batch dimension mismatch", func(t *testing.T) {
			loss := losses.NewMSE()

			yp, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Zeros([]int{2, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = loss.Compute(yp, yt)
			if err == nil {
				t.Fatal("expected error because of tensors having unequal batch sizes")
			} else if err.Error() != "MSE input data validation failed: expected input tensor sizes to match along batch dimension: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("MSE Compute with unequal data sizes / returns error: data dimension mismatch", func(t *testing.T) {
			loss := losses.NewMSE()

			yp, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Zeros([]int{1, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = loss.Compute(yp, yt)
			if err == nil {
				t.Fatal("expected error because of tensors having unequal data sizes")
			} else if err.Error() != "MSE input data validation failed: expected input tensor sizes to match along data dimension: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
