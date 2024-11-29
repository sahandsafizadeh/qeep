package initializers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestHeUniform(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(_ tensor.Device) {

		initializer, err := initializers.NewHeUniform(&initializers.HeUniformConfig{FanIn: 16})
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		x, err := initializer.Init([]int{32, 32})
		if err != nil {
			t.Fatal(err)
		}

		shape := x.Shape()
		if !shapesEqual(shape, []int{32, 32}) {
			t.Fatalf("expected tensor to have shape [32, 32], got %v", shape)
		}

		err = tensor.BackPropagate(x)
		if err != nil {
			t.Fatal(err)
		}

		if x.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

	})
}

func TestValidationHeUniform(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(_ tensor.Device) {

		/* ------------------------------ */

		_, err := initializers.NewHeUniform(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input config")
		} else if err.Error() != "HeUniform config data validation failed: expected config not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = initializers.NewHeUniform(&initializers.HeUniformConfig{FanIn: 0})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'FanIn'")
		} else if err.Error() != "HeUniform config data validation failed: expected 'FanIn' to be positive: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = initializers.NewHeUniform(&initializers.HeUniformConfig{FanIn: -1})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'FanIn'")
		} else if err.Error() != "HeUniform config data validation failed: expected 'FanIn' to be positive: got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
