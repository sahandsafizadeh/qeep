package initializers_test

import (
	"slices"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestUniform(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(_ tensor.Device) {

		initializer, err := initializers.NewUniform(nil)
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

func TestValidationUniform(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(_ tensor.Device) {

		/* ------------------------------ */

		_, err := initializers.NewUniform(&initializers.UniformConfig{
			Lower: 1.,
			Upper: 1.,
		})
		if err == nil {
			t.Fatalf("expected error because of 'Lower' not being less than 'Upper'")
		} else if err.Error() != "Uniform config data validation failed: expected 'Lower' to be less than 'Upper': (1.000000) >= (1.000000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = initializers.NewUniform(&initializers.UniformConfig{
			Lower: 1.5,
			Upper: 1.,
		})
		if err == nil {
			t.Fatalf("expected error because of 'Lower' not being less than 'Upper'")
		} else if err.Error() != "Uniform config data validation failed: expected 'Lower' to be less than 'Upper': (1.500000) >= (1.000000)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
