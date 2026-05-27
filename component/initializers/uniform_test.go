package initializers_test

import (
	"slices"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestUniform(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("NewUniform(nil) default-value initializer / Init([32,32]) / returns tensor with correct shape and non-nil gradient after backprop", func(t *testing.T) {
			initializer, err := initializers.NewUniform(nil)
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
				t.Fatalf("expected tensor to have shape [32, 32], got %v", shape)
			}

			if x.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		// ============================== validations ==============================

		t.Run("NewUniform(&UniformConfig{Lower: 1, Upper: 1}) / returns error: Lower not less than Upper (equal)", func(t *testing.T) {
			_, err := initializers.NewUniform(&initializers.UniformConfig{
				Lower: 1.,
				Upper: 1.,
			})
			if err == nil {
				t.Fatal("expected error because of 'Lower' not being less than 'Upper'")
			} else if err.Error() != "Uniform config data validation failed: expected 'Lower' to be less than 'Upper': (1.000000) >= (1.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewUniform(&UniformConfig{Lower: 1.5, Upper: 1}) / returns error: Lower not less than Upper (greater)", func(t *testing.T) {
			_, err := initializers.NewUniform(&initializers.UniformConfig{
				Lower: 1.5,
				Upper: 1.,
			})
			if err == nil {
				t.Fatal("expected error because of 'Lower' not being less than 'Upper'")
			} else if err.Error() != "Uniform config data validation failed: expected 'Lower' to be less than 'Upper': (1.500000) >= (1.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
