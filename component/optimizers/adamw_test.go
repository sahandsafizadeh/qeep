package optimizers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/optimizers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestAdamW(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("AdamW lr=2 wd=1 beta1=0.5 beta2=0.75 eps=100, 32x32 tensor x=2 / 3 consecutive update steps with grad 100 / x oscillates: 2→-3→2→-3", func(t *testing.T) {
			optimizer, err := optimizers.NewAdamW(&optimizers.AdamWConfig{
				LearningRate: 2.,
				WeightDecay:  1.,
				Beta1:        0.5,
				Beta2:        0.75,
				Eps:          100.,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Full([]int{32, 32}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			// step 1: x = 2 → -3
			y := x.Scale(4.).Scale(5.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{32, 32}, -3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal after step 1")
			}

			// step 2: x = -3 → 2
			x.ResetGradContext(true)
			y = x.Scale(4.).Scale(5.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err = tensor.Full([]int{32, 32}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal after step 2")
			}

			// step 3: x = 2 → -3
			x.ResetGradContext(true)
			y = x.Scale(4.).Scale(5.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err = tensor.Full([]int{32, 32}, -3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal after step 3")
			}
		})

		t.Run("AdamW lr=2 wd=1 beta1=0.5 beta2=0.75 eps=100, 16x16 tensor x=3 / 3 consecutive update steps with grad 100 / x oscillates: 3→-4→3→-4", func(t *testing.T) {
			optimizer, err := optimizers.NewAdamW(&optimizers.AdamWConfig{
				LearningRate: 2.,
				WeightDecay:  1.,
				Beta1:        0.5,
				Beta2:        0.75,
				Eps:          100.,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Full([]int{16, 16}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			// step 1: x = 3 → -4
			y := x.Scale(4.).Scale(5.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{16, 16}, -4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal after step 1")
			}

			// step 2: x = -4 → 3
			x.ResetGradContext(true)
			y = x.Scale(4.).Scale(5.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err = tensor.Full([]int{16, 16}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal after step 2")
			}

			// step 3: x = 3 → -4
			x.ResetGradContext(true)
			y = x.Scale(4.).Scale(5.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err = tensor.Full([]int{16, 16}, -4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal after step 3")
			}
		})

		t.Run("AdamW nil config (defaults), scalar x=1 / 2 consecutive update steps with grad 100 / x converges: ~0.99899→~0.99798", func(t *testing.T) {
			optimizer, err := optimizers.NewAdamW(nil)
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			// step 1: x ≈ 0.99899
			y := x.Scale(4.).Scale(5.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Full(nil, 0.99899-1e-5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Full(nil, 0.99899+1e-5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := x.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := x.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}

			// step 2: x ≈ 0.99798
			x.ResetGradContext(true)
			y = x.Scale(4.).Scale(5.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err = tensor.Full(nil, 0.99798-1e-5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err = tensor.Full(nil, 0.99798+1e-5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := x.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := x.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		// ============================== validations ==============================

		t.Run("NewAdamW with LearningRate=0 / returns error: non-positive LearningRate", func(t *testing.T) {
			_, err := optimizers.NewAdamW(&optimizers.AdamWConfig{})
			if err == nil {
				t.Fatal("expected error because of non-positive 'LearningRate'")
			} else if err.Error() != "AdamW config data validation failed: expected 'LearningRate' to be positive: got (0.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewAdamW with LearningRate=-1 / returns error: non-positive LearningRate", func(t *testing.T) {
			_, err := optimizers.NewAdamW(&optimizers.AdamWConfig{
				LearningRate: -1,
			})
			if err == nil {
				t.Fatal("expected error because of non-positive 'LearningRate'")
			} else if err.Error() != "AdamW config data validation failed: expected 'LearningRate' to be positive: got (-1.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewAdamW with WeightDecay=-1 / returns error: negative WeightDecay", func(t *testing.T) {
			_, err := optimizers.NewAdamW(&optimizers.AdamWConfig{
				LearningRate: 1,
				WeightDecay:  -1,
			})
			if err == nil {
				t.Fatal("expected error because of negative 'WeightDecay'")
			} else if err.Error() != "AdamW config data validation failed: expected 'WeightDecay' not to be negative: got (-1.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewAdamW with Beta1=-0.5 / returns error: negative Beta1", func(t *testing.T) {
			_, err := optimizers.NewAdamW(&optimizers.AdamWConfig{
				LearningRate: 10,
				WeightDecay:  0,
				Beta1:        -0.5,
			})
			if err == nil {
				t.Fatal("expected error because of negative 'Beta1'")
			} else if err.Error() != "AdamW config data validation failed: expected 'Beta1' not to be negative: got (-0.500000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewAdamW with Beta2=-0.1 / returns error: negative Beta2", func(t *testing.T) {
			_, err := optimizers.NewAdamW(&optimizers.AdamWConfig{
				LearningRate: 10,
				WeightDecay:  1,
				Beta1:        0,
				Beta2:        -0.1,
			})
			if err == nil {
				t.Fatal("expected error because of negative 'Beta2'")
			} else if err.Error() != "AdamW config data validation failed: expected 'Beta2' not to be negative: got (-0.100000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewAdamW with Eps=0 / returns error: non-positive Eps", func(t *testing.T) {
			_, err := optimizers.NewAdamW(&optimizers.AdamWConfig{
				LearningRate: 100,
				WeightDecay:  10,
				Beta1:        1,
				Beta2:        0,
			})
			if err == nil {
				t.Fatal("expected error because of non-positive 'Eps'")
			} else if err.Error() != "AdamW config data validation failed: expected 'Eps' to be positive: got (0.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewAdamW with Eps=-1 / returns error: non-positive Eps", func(t *testing.T) {
			_, err := optimizers.NewAdamW(&optimizers.AdamWConfig{
				LearningRate: 100,
				WeightDecay:  10,
				Beta1:        1,
				Beta2:        0.5,
				Eps:          -1,
			})
			if err == nil {
				t.Fatal("expected error because of non-positive 'Eps'")
			} else if err.Error() != "AdamW config data validation failed: expected 'Eps' to be positive: got (-1.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Update on nil tensor / returns error: nil tensor", func(t *testing.T) {
			optimizer, err := optimizers.NewAdamW(nil)
			if err != nil {
				t.Fatal(err)
			}

			var x tensor.Tensor

			err = optimizer.Update(&x)
			if err == nil {
				t.Fatal("expected error because of nil tensor")
			} else if err.Error() != "AdamW input data validation failed: expected optimized tensor not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Update on tensor without gradient tracking / returns error: nil gradient", func(t *testing.T) {
			optimizer, err := optimizers.NewAdamW(nil)
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = optimizer.Update(&x)
			if err == nil {
				t.Fatal("expected error because of nil tensor gradient")
			} else if err.Error() != "AdamW input data validation failed: expected tensor's gradient not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Update on tracked tensor before backward pass / returns error: nil gradient", func(t *testing.T) {
			optimizer, err := optimizers.NewAdamW(nil)
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			err = optimizer.Update(&x)
			if err == nil {
				t.Fatal("expected error because of nil tensor gradient")
			} else if err.Error() != "AdamW input data validation failed: expected tensor's gradient not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Update on untracked tensor after backward pass / returns error: nil gradient", func(t *testing.T) {
			optimizer, err := optimizers.NewAdamW(nil)
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(x)
			if err != nil {
				t.Fatal(err)
			}

			err = optimizer.Update(&x)
			if err == nil {
				t.Fatal("expected error because of nil tensor gradient")
			} else if err.Error() != "AdamW input data validation failed: expected tensor's gradient not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
