package optimizers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/optimizers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestSGD(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("SGD lr=0.1, x=5 / one update step with gradient 30 / x becomes 2", func(t *testing.T) {
			optimizer, err := optimizers.NewSGD(&optimizers.SGDConfig{
				LearningRate: 0.1,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Full([]int{32, 32}, 5., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(2.).Scale(3.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{32, 32}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("SGD lr=0.1, x=2 / one update step with gradient 20 / x becomes 0", func(t *testing.T) {
			optimizer, err := optimizers.NewSGD(&optimizers.SGDConfig{
				LearningRate: 0.1,
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

			y := x.Scale(2.).Scale(2.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{32, 32}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("SGD lr=0.1 momentum=0.5, x=15 / 3 consecutive update steps / velocity accumulates: x reaches 1.5", func(t *testing.T) {
			optimizer, err := optimizers.NewSGD(&optimizers.SGDConfig{
				LearningRate: 0.1,
				Momentum:     0.5,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Full([]int{32, 32}, 15., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			// step 1: grad=40, v=40, x = 15 - 0.1*40 = 11
			y := x.Scale(2.).Scale(4.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{32, 32}, 11., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal after step 1")
			}

			// step 2: grad=30, v=0.5*40+30=50, x = 11 - 0.1*50 = 6
			x.ResetGradContext(true)
			y = x.Scale(2.).Scale(3.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err = tensor.Full([]int{32, 32}, 6., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal after step 2")
			}

			// step 3: grad=20, v=0.5*50+20=45, x = 6 - 0.1*45 = 1.5
			x.ResetGradContext(true)
			y = x.Scale(2.).Scale(2.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err = tensor.Full([]int{32, 32}, 1.5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal after step 3")
			}
		})

		t.Run("SGD lr=0.1 weightDecay=0.2 momentum=0.5, x=100 / 3 consecutive update steps / applies decay and momentum: x reaches 78.4576", func(t *testing.T) {
			optimizer, err := optimizers.NewSGD(&optimizers.SGDConfig{
				LearningRate: 0.1,
				WeightDecay:  0.2,
				Momentum:     0.5,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Full([]int{32, 32}, 100., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			// step 1: grad=40, delta=40+0.2*100=60, v=60, x = 100 - 0.1*60 = 94
			y := x.Scale(2.).Scale(4.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{32, 32}, 94., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal after step 1")
			}

			// step 2: grad=30, delta=30+0.2*94+0.5*60=78.8, x = 94 - 0.1*78.8 = 86.12
			x.ResetGradContext(true)
			y = x.Scale(2.).Scale(3.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err = tensor.Full([]int{32, 32}, 86.12, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal after step 2")
			}

			// step 3: grad=20, delta=20+0.2*86.12+0.5*78.8=76.624, x = 86.12 - 0.1*76.624 = 78.4576
			x.ResetGradContext(true)
			y = x.Scale(2.).Scale(2.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}
			err = optimizer.Update(&x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err = tensor.Full([]int{32, 32}, 78.4576, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal after step 3")
			}
		})

		// ============================== validations ==============================

		t.Run("NewSGD with LearningRate=0 / returns error: non-positive LearningRate", func(t *testing.T) {
			_, err := optimizers.NewSGD(&optimizers.SGDConfig{})
			if err == nil {
				t.Fatal("expected error because of non-positive 'LearningRate'")
			} else if err.Error() != "SGD config data validation failed: expected 'LearningRate' to be positive: got (0.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewSGD with LearningRate=-1 / returns error: non-positive LearningRate", func(t *testing.T) {
			_, err := optimizers.NewSGD(&optimizers.SGDConfig{
				LearningRate: -1,
			})
			if err == nil {
				t.Fatal("expected error because of non-positive 'LearningRate'")
			} else if err.Error() != "SGD config data validation failed: expected 'LearningRate' to be positive: got (-1.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewSGD with WeightDecay=-1 / returns error: negative WeightDecay", func(t *testing.T) {
			_, err := optimizers.NewSGD(&optimizers.SGDConfig{
				LearningRate: 1,
				WeightDecay:  -1,
			})
			if err == nil {
				t.Fatal("expected error because of negative 'WeightDecay'")
			} else if err.Error() != "SGD config data validation failed: expected 'WeightDecay' not to be negative: got (-1.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewSGD with Momentum=-0.5 / returns error: negative Momentum", func(t *testing.T) {
			_, err := optimizers.NewSGD(&optimizers.SGDConfig{
				LearningRate: 10,
				WeightDecay:  1,
				Momentum:     -0.5,
			})
			if err == nil {
				t.Fatal("expected error because of negative 'Momentum'")
			} else if err.Error() != "SGD config data validation failed: expected 'Momentum' not to be negative: got (-0.500000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Update on tensor without gradient tracking / returns error: nil gradient", func(t *testing.T) {
			optimizer, err := optimizers.NewSGD(nil)
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
			} else if err.Error() != "SGD input data validation failed: expected tensor's gradient not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Update on tracked tensor before backward pass / returns error: nil gradient", func(t *testing.T) {
			optimizer, err := optimizers.NewSGD(nil)
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
			} else if err.Error() != "SGD input data validation failed: expected tensor's gradient not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Update on untracked tensor after backward pass / returns error: nil gradient", func(t *testing.T) {
			optimizer, err := optimizers.NewSGD(nil)
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
			} else if err.Error() != "SGD input data validation failed: expected tensor's gradient not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
