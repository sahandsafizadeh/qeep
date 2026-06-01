package initializers_test

import (
	"slices"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestHeUniform(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("NewHeUniform(&HeUniformConfig{FanIn: 16}) / Init([32,32]) / returns tensor with correct shape and non-nil gradient after backprop", func(t *testing.T) {
			initializer, err := initializers.NewHeUniform(&initializers.HeUniformConfig{FanIn: 16})
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

		t.Run("NewHeUniform(nil) / returns error: nil config", func(t *testing.T) {
			_, err := initializers.NewHeUniform(nil)
			if err == nil {
				t.Fatal("expected error because of nil input config")
			} else if err.Error() != "HeUniform config data validation failed: expected config not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewHeUniform(&HeUniformConfig{FanIn: 0}) / returns error: non-positive FanIn", func(t *testing.T) {
			_, err := initializers.NewHeUniform(&initializers.HeUniformConfig{FanIn: 0})
			if err == nil {
				t.Fatal("expected error because of non-positive 'FanIn'")
			} else if err.Error() != "HeUniform config data validation failed: expected 'FanIn' to be positive: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewHeUniform(&HeUniformConfig{FanIn: -1}) / returns error: non-positive FanIn", func(t *testing.T) {
			_, err := initializers.NewHeUniform(&initializers.HeUniformConfig{FanIn: -1})
			if err == nil {
				t.Fatal("expected error because of non-positive 'FanIn'")
			} else if err.Error() != "HeUniform config data validation failed: expected 'FanIn' to be positive: got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
