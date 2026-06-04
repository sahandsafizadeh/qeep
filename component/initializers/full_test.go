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

		t.Run("NewFull(nil) / Init([1,1]) / returns tensor equal to zero tensor", func(t *testing.T) {
			initializer := initializers.NewFull(nil)

			act, err := initializer.Init([]int{1, 1}, dev)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{1, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("NewFull(&FullConfig{}) / Init([1,1]) / returns tensor equal to zero tensor", func(t *testing.T) {
			initializer := initializers.NewFull(&initializers.FullConfig{})

			act, err := initializer.Init([]int{1, 1}, dev)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{1, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

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
	})
}
