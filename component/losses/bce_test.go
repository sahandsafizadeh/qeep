package losses_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestBCE(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("BCE / Compute([[0]] yp, [[0]] yt) / output near 0", func(t *testing.T) {
			loss := losses.NewBCE()

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
				t.Fatalf("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatalf("expected output to be in range")
			}
		})

		t.Run("BCE / Compute([[1]] yp, [[1]] yt) / output near 0", func(t *testing.T) {
			loss := losses.NewBCE()

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
				t.Fatalf("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatalf("expected output to be in range")
			}
		})

		t.Run("BCE / Compute([[0]] yp, [[2]] yt) / output near 27.631", func(t *testing.T) {
			loss := losses.NewBCE()

			yp, err := tensor.Of([][]float64{{0.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{{2.}}, &tensor.Config{Device: dev})
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
				t.Fatalf("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatalf("expected output to be in range")
			}
		})

		t.Run("BCE / Compute([[1.5]] yp, [[-1]] yt) / output near 27.631", func(t *testing.T) {
			loss := losses.NewBCE()

			yp, err := tensor.Of([][]float64{{1.5}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{{-1.}}, &tensor.Config{Device: dev})
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
				t.Fatalf("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatalf("expected output to be in range")
			}
		})

		t.Run("BCE / Compute([[0.1],[0.2],[0.8],[0.9]] yp, [[0],[0],[1],[1]] yt) / output near 0.164", func(t *testing.T) {
			loss := losses.NewBCE()

			yp, err := tensor.Of([][]float64{{0.1}, {0.2}, {0.8}, {0.9}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{{0.}, {0.}, {1.}, {1.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := loss.Compute(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of(0.163, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of(0.165, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatalf("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatalf("expected output to be in range")
			}
		})

		t.Run("BCE / Compute([[0.9],[0.8],[0.2],[0.1]] yp, [[0],[0],[1],[1]] yt) / output near 1.956", func(t *testing.T) {
			loss := losses.NewBCE()

			yp, err := tensor.Of([][]float64{{0.9}, {0.8}, {0.2}, {0.1}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Of([][]float64{{0.}, {0.}, {1.}, {1.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := loss.Compute(yp, yt)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of(1.955, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of(1.957, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatalf("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatalf("expected output to be in range")
			}
		})

		// ============================== validations ==============================

		t.Run("BCE Compute with yp rank-1 and yt rank-2 / returns error: expected exactly two dimensions", func(t *testing.T) {
			loss := losses.NewBCE()

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
				t.Fatalf("expected error because of tensors having more/less than two dimensions")
			} else if err.Error() != "BCE input data validation failed: expected input tensors to have exactly two dimensions (batch, class=1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("BCE Compute with yp rank-2 and yt rank-3 / returns error: expected exactly two dimensions", func(t *testing.T) {
			loss := losses.NewBCE()

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
				t.Fatalf("expected error because of tensors having more/less than two dimensions")
			} else if err.Error() != "BCE input data validation failed: expected input tensors to have exactly two dimensions (batch, class=1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("BCE Compute with unequal batch sizes / returns error: batch dimension mismatch", func(t *testing.T) {
			loss := losses.NewBCE()

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
				t.Fatalf("expected error because of tensors having unequal batch sizes")
			} else if err.Error() != "BCE input data validation failed: expected input tensor sizes to match along batch dimension: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("BCE Compute with yp class size 1 and yt class size 2 / returns error: class dimension not equal to 1", func(t *testing.T) {
			loss := losses.NewBCE()

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
				t.Fatalf("expected error because of tensors having class sizes unequal to (1)")
			} else if err.Error() != "BCE input data validation failed: expected input tensor sizes to be equal to (1) along class dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("BCE Compute with yp class size 2 and yt class size 1 / returns error: class dimension not equal to 1", func(t *testing.T) {
			loss := losses.NewBCE()

			yp, err := tensor.Zeros([]int{1, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			yt, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = loss.Compute(yp, yt)
			if err == nil {
				t.Fatalf("expected error because of tensors having class sizes unequal to (1)")
			} else if err.Error() != "BCE input data validation failed: expected input tensor sizes to be equal to (1) along class dimension" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
