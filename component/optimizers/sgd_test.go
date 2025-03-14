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

		optimizer, err := optimizers.NewSGD(&optimizers.SGDConfig{
			LearningRate: 0.1,
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err := tensor.Full([]int{32, 32}, 5., conf)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		x.ResetGradContext(true)

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

		exp, err := tensor.Full([]int{32, 32}, 2., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		x.ResetGradContext(true)

		y = x.Scale(2.).Scale(2.).Scale(5.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x)
		if err != nil {
			t.Fatal(err)
		}

		act = x

		exp, err = tensor.Full([]int{32, 32}, 0., conf)
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

func TestSGDWithMomentum(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{
			Device:    dev,
			GradTrack: true,
		}

		optimizer, err := optimizers.NewSGD(&optimizers.SGDConfig{
			LearningRate: 0.1,
			Momentum:     0.5,
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err := tensor.Full([]int{32, 32}, 15., conf)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		x.ResetGradContext(true)

		y := x.Scale(2.).Scale(4.).Scale(5.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x)
		if err != nil {
			t.Fatal(err)
		}

		act := x

		exp, err := tensor.Full([]int{32, 32}, 11., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		x.ResetGradContext(true)

		y = x.Scale(2.).Scale(3.).Scale(5.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x)
		if err != nil {
			t.Fatal(err)
		}

		act = x

		exp, err = tensor.Full([]int{32, 32}, 6., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		x.ResetGradContext(true)

		y = x.Scale(2.).Scale(2.).Scale(5.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x)
		if err != nil {
			t.Fatal(err)
		}

		act = x

		exp, err = tensor.Full([]int{32, 32}, 1.5, conf)
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

func TestSGDWithMomentumAndWeightDecay(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{
			Device:    dev,
			GradTrack: true,
		}

		optimizer, err := optimizers.NewSGD(&optimizers.SGDConfig{
			LearningRate: 0.1,
			WeightDecay:  0.2,
			Momentum:     0.5,
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err := tensor.Full([]int{32, 32}, 100., conf)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		x.ResetGradContext(true)

		y := x.Scale(2.).Scale(4.).Scale(5.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x)
		if err != nil {
			t.Fatal(err)
		}

		act := x

		exp, err := tensor.Full([]int{32, 32}, 94., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		x.ResetGradContext(true)

		y = x.Scale(2.).Scale(3.).Scale(5.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x)
		if err != nil {
			t.Fatal(err)
		}

		act = x

		exp, err = tensor.Full([]int{32, 32}, 86.12, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		x.ResetGradContext(true)

		y = x.Scale(2.).Scale(2.).Scale(5.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x)
		if err != nil {
			t.Fatal(err)
		}

		act = x

		exp, err = tensor.Full([]int{32, 32}, 78.4576, conf)
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

		/* ------------------------------ */

		_, err := optimizers.NewSGD(&optimizers.SGDConfig{})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'LearningRate'")
		} else if err.Error() != "SGD config data validation failed: expected 'LearningRate' to be positive: got (0.000000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = optimizers.NewSGD(&optimizers.SGDConfig{
			LearningRate: -1,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'LearningRate'")
		} else if err.Error() != "SGD config data validation failed: expected 'LearningRate' to be positive: got (-1.000000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = optimizers.NewSGD(&optimizers.SGDConfig{
			LearningRate: 1,
			WeightDecay:  -1,
		})
		if err == nil {
			t.Fatalf("expected error because of negative 'WeightDecay'")
		} else if err.Error() != "SGD config data validation failed: expected 'WeightDecay' not to be negative: got (-1.000000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = optimizers.NewSGD(&optimizers.SGDConfig{
			LearningRate: 10,
			WeightDecay:  1,
			Momentum:     -0.5,
		})
		if err == nil {
			t.Fatalf("expected error because of negative 'Momentum'")
		} else if err.Error() != "SGD config data validation failed: expected 'Momentum' not to be negative: got (-0.500000)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		optimizer, err := optimizers.NewSGD(nil)
		if err != nil {
			t.Fatal(err)
		}

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
