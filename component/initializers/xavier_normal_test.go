package initializers_test

import (
	"slices"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestXavierNormal(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("NewXavierNormal(&XavierNormalConfig{FanIn: 16, FanOut: 4}) / Init([32,32]) / returns tensor with correct shape and non-nil gradient after backprop", func(t *testing.T) {
			initializer, err := initializers.NewXavierNormal(&initializers.XavierNormalConfig{
				FanIn:  16,
				FanOut: 4,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := initializer.Init([]int{32, 32}, dev)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(x)
			if err != nil {
				t.Fatal(err)
			}

			if shape := x.Shape(); !slices.Equal(shape, []int{32, 32}) {
				t.Fatal("expected tensor to have shape [32, 32], got", shape)
			}

			if x.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		// ============================== validations ==============================

		t.Run("NewXavierNormal(nil) / returns error: nil config", func(t *testing.T) {
			_, err := initializers.NewXavierNormal(nil)
			if err == nil {
				t.Fatal("expected error because of nil input config")
			} else if err.Error() != "XavierNormal config data validation failed: expected config not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewXavierNormal(&XavierNormalConfig{FanIn: 0, FanOut: 1}) / returns error: non-positive FanIn", func(t *testing.T) {
			_, err := initializers.NewXavierNormal(&initializers.XavierNormalConfig{
				FanIn:  0,
				FanOut: 1,
			})
			if err == nil {
				t.Fatal("expected error because of non-positive 'FanIn'")
			} else if err.Error() != "XavierNormal config data validation failed: expected 'FanIn' to be positive: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewXavierNormal(&XavierNormalConfig{FanIn: -1, FanOut: 1}) / returns error: non-positive FanIn", func(t *testing.T) {
			_, err := initializers.NewXavierNormal(&initializers.XavierNormalConfig{
				FanIn:  -1,
				FanOut: 1,
			})
			if err == nil {
				t.Fatal("expected error because of non-positive 'FanIn'")
			} else if err.Error() != "XavierNormal config data validation failed: expected 'FanIn' to be positive: got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewXavierNormal(&XavierNormalConfig{FanIn: 1, FanOut: 0}) / returns error: non-positive FanOut", func(t *testing.T) {
			_, err := initializers.NewXavierNormal(&initializers.XavierNormalConfig{
				FanIn:  1,
				FanOut: 0,
			})
			if err == nil {
				t.Fatal("expected error because of non-positive 'FanOut'")
			} else if err.Error() != "XavierNormal config data validation failed: expected 'FanOut' to be positive: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewXavierNormal(&XavierNormalConfig{FanIn: 1, FanOut: -1}) / returns error: non-positive FanOut", func(t *testing.T) {
			_, err := initializers.NewXavierNormal(&initializers.XavierNormalConfig{
				FanIn:  1,
				FanOut: -1,
			})
			if err == nil {
				t.Fatal("expected error because of non-positive 'FanOut'")
			} else if err.Error() != "XavierNormal config data validation failed: expected 'FanOut' to be positive: got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
