package optimizers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/optimizers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestSGD(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{
			Device:    dev,
			GradTrack: true,
		}

		optimizer := optimizers.NewSGD(&optimizers.SGDConfig{LearningRate: 0.1})

		/* ------------------------------ */

		x, err := tensor.Full([]int{32, 32}, 4., conf)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Scale(2.).Scale(3.).Scale(5.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x)
		if err != nil {
			t.Fatal(err)
		}

		act := x

		exp, err := tensor.Full([]int{32, 32}, 1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestValidationSGD(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		confU := &tensor.Config{
			Device:    dev,
			GradTrack: false,
		}

		confT := &tensor.Config{
			Device:    dev,
			GradTrack: true,
		}

		optimizer := optimizers.NewSGD(nil)

		/* ------------------------------ */

		wu, err := tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		wt, err := tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&wu)
		if err == nil {
			t.Fatalf("expected error because of nil tensor gradient")
		} else if err.Error() != "SGD input data validation failed: expected tensor's gradient not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		err = optimizer.Update(&wt)
		if err == nil {
			t.Fatalf("expected error because of nil tensor gradient")
		} else if err.Error() != "SGD input data validation failed: expected tensor's gradient not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		err = tensor.BackPropagate(wu)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&wu)
		if err == nil {
			t.Fatalf("expected error because of nil tensor gradient")
		} else if err.Error() != "SGD input data validation failed: expected tensor's gradient not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
