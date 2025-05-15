package initializers_test

import (
	"slices"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestXavierNormal(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		initializer, err := initializers.NewXavierNormal(&initializers.XavierNormalConfig{
			FanIn:  16,
			FanOut: 4,
		})
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		x, err := initializer.Init([]int{32, 32}, dev)
		if err != nil {
			t.Fatal(err)
		}

		shape := x.Shape()
		if !slices.Equal(shape, []int{32, 32}) {
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

func TestValidationXavierNormal(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(_ tensor.Device) {

		/* ------------------------------ */

		_, err := initializers.NewXavierNormal(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input config")
		} else if err.Error() != "XavierNormal config data validation failed: expected config not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = initializers.NewXavierNormal(&initializers.XavierNormalConfig{
			FanIn:  0,
			FanOut: 1,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'FanIn'")
		} else if err.Error() != "XavierNormal config data validation failed: expected 'FanIn' to be positive: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = initializers.NewXavierNormal(&initializers.XavierNormalConfig{
			FanIn:  -1,
			FanOut: 1,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'FanIn'")
		} else if err.Error() != "XavierNormal config data validation failed: expected 'FanIn' to be positive: got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = initializers.NewXavierNormal(&initializers.XavierNormalConfig{
			FanIn:  1,
			FanOut: 0,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'FanOut'")
		} else if err.Error() != "XavierNormal config data validation failed: expected 'FanOut' to be positive: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = initializers.NewXavierNormal(&initializers.XavierNormalConfig{
			FanIn:  1,
			FanOut: -1,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'FanOut'")
		} else if err.Error() != "XavierNormal config data validation failed: expected 'FanOut' to be positive: got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
