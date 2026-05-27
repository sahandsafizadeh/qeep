package losses_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestCE(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("CE loss / Compute([[0]], [[0]]) / output near 0", func(t *testing.T) {
			loss := losses.NewCE()

			yp, err := tensor.Of([][]float64{{0.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{{0.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := loss.Compute(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of(0.-1e-10, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of(0.+1e-10, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("CE loss / Compute([[1]], [[1]]) / output near 0", func(t *testing.T) {
			loss := losses.NewCE()

			yp, err := tensor.Of([][]float64{{1.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{{1.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := loss.Compute(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of(0.-1e-10, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of(0.+1e-10, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("CE loss / Compute([[0, 1.5]], [[2, -1]]) / output near 27.631", func(t *testing.T) {
			loss := losses.NewCE()

			yp, err := tensor.Of([][]float64{{0., 1.5}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{{2., -1.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := loss.Compute(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of(27.630, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of(27.632, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("CE loss / Compute(3x3 near-correct predictions, 3x3 one-hot) / output near 0.228", func(t *testing.T) {
			loss := losses.NewCE()

			yp, err := tensor.Of([][]float64{
				{0.1, 0.8, 0.1},
				{0.1, 0.2, 0.7},
				{0.9, 0.1, 0.0},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{
				{0., 1., 0.},
				{0., 0., 1.},
				{1., 0., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := loss.Compute(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of(0.227, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of(0.229, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("CE loss / Compute(3x3 spread predictions, 3x3 one-hot) / output near 1.705", func(t *testing.T) {
			loss := losses.NewCE()

			yp, err := tensor.Of([][]float64{
				{0.4, 0.1, 0.5},
				{0.4, 0.4, 0.2},
				{0.3, 0.3, 0.4},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{
				{0., 1., 0.},
				{0., 0., 1.},
				{1., 0., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := loss.Compute(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of(1.704, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of(1.706, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		// ============================== validations ==============================

		t.Run("CE.Compute(1D tensor, 2D tensor) / returns error: expected exactly two dimensions", func(t *testing.T) {
			loss := losses.NewCE()

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
			} else if err.Error() != "CE input data validation failed: expected input tensors to have exactly two dimensions (batch, class)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("CE.Compute(2D tensor, 3D tensor) / returns error: expected exactly two dimensions", func(t *testing.T) {
			loss := losses.NewCE()

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
			} else if err.Error() != "CE input data validation failed: expected input tensors to have exactly two dimensions (batch, class)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("CE.Compute([1x1], [2x1]) / returns error: batch sizes do not match", func(t *testing.T) {
			loss := losses.NewCE()

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
			} else if err.Error() != "CE input data validation failed: expected input tensor sizes to match along batch dimension: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("CE.Compute([1x1], [1x2]) / returns error: class sizes do not match", func(t *testing.T) {
			loss := losses.NewCE()

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
				t.Fatal("expected error because of tensors having unequal class sizes")
			} else if err.Error() != "CE input data validation failed: expected input tensor sizes to match along class dimension: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
