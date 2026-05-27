package initializers_test

import (
	"slices"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestFull(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("NewFull(nil) default-value initializer / Init([32,32]) / returns tensor with correct shape and non-nil gradient after backprop", func(t *testing.T) {
			initializer := initializers.NewFull(nil)

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
	})
}
