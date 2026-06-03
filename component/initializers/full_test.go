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

		t.Run("NewFull(Value=-1) / Init([]) / returns scalar tensor with value -1", func(t *testing.T) {
			initializer := initializers.NewFull(&initializers.FullConfig{Value: -1.})

			x, err := initializer.Init(nil, dev)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(-1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
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

		t.Run("NewFull(nil) / Init([]) / returns scalar tensor with default value 0", func(t *testing.T) {
			initializer := initializers.NewFull(nil)

			x, err := initializer.Init(nil, dev)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})
	})
}
