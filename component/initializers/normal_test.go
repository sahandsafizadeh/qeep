package initializers_test

import (
	"slices"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestNormal(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(_ tensor.Device) {

		initializer, err := initializers.NewNormal(nil)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		x, err := initializer.Init([]int{32, 32})
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

func TestValidationNormal(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(_ tensor.Device) {

		/* ------------------------------ */

		_, err := initializers.NewNormal(&initializers.NormalConfig{StdDev: 0.})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'StdDev'")
		} else if err.Error() != "Normal config data validation failed: expected 'StdDev' to be positive: got (0.000000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = initializers.NewNormal(&initializers.NormalConfig{StdDev: -1.})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'StdDev'")
		} else if err.Error() != "Normal config data validation failed: expected 'StdDev' to be positive: got (-1.000000)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
