package initializers_test

import (
	"slices"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestNormal(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("NewNormal(nil) default-value initializer / Init([32,32]) / returns tensor with correct shape and non-nil gradient after backprop", func(t *testing.T) {
			initializer, err := initializers.NewNormal(nil)
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
				t.Fatalf("expected gradient not to be nil")
			}
		})

		// ============================== validations ==============================

		t.Run("NewNormal(&NormalConfig{StdDev: 0}) / returns error: non-positive StdDev", func(t *testing.T) {
			_, err := initializers.NewNormal(&initializers.NormalConfig{StdDev: 0.})
			if err == nil {
				t.Fatalf("expected error because of non-positive 'StdDev'")
			} else if err.Error() != "Normal config data validation failed: expected 'StdDev' to be positive: got (0.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewNormal(&NormalConfig{StdDev: negative}) / returns error: non-positive StdDev", func(t *testing.T) {
			_, err := initializers.NewNormal(&initializers.NormalConfig{StdDev: -1.})
			if err == nil {
				t.Fatalf("expected error because of non-positive 'StdDev'")
			} else if err.Error() != "Normal config data validation failed: expected 'StdDev' to be positive: got (-1.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
