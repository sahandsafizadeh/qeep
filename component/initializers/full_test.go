package initializers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestFull(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(_ tensor.Device) {

		initializer := initializers.NewFull(nil)

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

/* ----- helpers ----- */

func shapesEqual(s1, s2 []int) (ok bool) {
	if len(s1) != len(s2) {
		return false
	}

	for i := 0; i < len(s1); i++ {
		if s1[i] != s2[i] {
			return false
		}
	}

	return true
}
