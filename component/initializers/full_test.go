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

		t.Run("NewFull(Value=5) / Init([32,32]) / returns tensor with correct shape and non-nil gradient after backprop", func(t *testing.T) {
			initializer := initializers.NewFull(&initializers.FullConfig{Value: 5.})

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

		t.Run("NewFull(nil) / NewFull(&FullConfig{}) / both use FullDefaultValue and produce equal tensors", func(t *testing.T) {
			initializer1 := initializers.NewFull(nil)
			initializer2 := initializers.NewFull(&initializers.FullConfig{})

			x1, err := initializer1.Init(nil, dev)
			if err != nil {
				t.Fatal(err)
			}
			x2, err := initializer2.Init(nil, dev)
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x1.Equals(x2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})
	})
}
