package initializers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestHeNormal(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(_ tensor.Device) {

		initializer, err := initializers.NewHeNormal(&initializers.HeNormalConfig{FanIn: 16})
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

func TestValidationHeNormal(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(_ tensor.Device) {

		/* ------------------------------ */

		_, err := initializers.NewHeNormal(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input config")
		} else if err.Error() != "HeNormal config data validation failed: expected config not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = initializers.NewHeNormal(&initializers.HeNormalConfig{FanIn: 0})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'FanIn'")
		} else if err.Error() != "HeNormal config data validation failed: expected 'FanIn' to be positive: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = initializers.NewHeNormal(&initializers.HeNormalConfig{FanIn: -1})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'FanIn'")
		} else if err.Error() != "HeNormal config data validation failed: expected 'FanIn' to be positive: got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
