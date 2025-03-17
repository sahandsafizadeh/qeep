package optimizers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/optimizers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestAdam(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{
			Device:    dev,
			GradTrack: true,
		}

		optimizer, err := optimizers.NewAdam(&optimizers.AdamConfig{
			LearningRate: 2.,
			Beta1:        0.5,
			Beta2:        0.75,
			Eps:          100.,
		})
		if err != nil {
			t.Fatal(err)
		}

		x1, err := tensor.Full([]int{32, 32}, 2., conf)
		if err != nil {
			t.Fatal(err)
		}

		x2, err := tensor.Full([]int{16, 16}, 3., conf)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		x1.ResetGradContext(true)
		x2.ResetGradContext(true)

		y1 := x1.Scale(4.).Scale(5.).Scale(5.)
		y2 := x2.Scale(4.).Scale(5.).Scale(5.)

		err = tensor.BackPropagate(y1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y2)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x1)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x2)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x1

		exp, err := tensor.Full([]int{32, 32}, 1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act = x2

		exp, err = tensor.Full([]int{16, 16}, 2., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		x1.ResetGradContext(true)
		x2.ResetGradContext(true)

		y1 = x1.Scale(4.).Scale(5.).Scale(5.)
		y2 = x2.Scale(4.).Scale(5.).Scale(5.)

		err = tensor.BackPropagate(y1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y2)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x1)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x2)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act = x1

		exp, err = tensor.Full([]int{32, 32}, 0., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act = x2

		exp, err = tensor.Full([]int{16, 16}, 1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		x1.ResetGradContext(true)
		x2.ResetGradContext(true)

		y1 = x1.Scale(4.).Scale(5.).Scale(5.)
		y2 = x2.Scale(4.).Scale(5.).Scale(5.)

		err = tensor.BackPropagate(y1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y2)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x1)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x2)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act = x1

		exp, err = tensor.Full([]int{32, 32}, -1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act = x2

		exp, err = tensor.Full([]int{16, 16}, 0., conf)
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

func TestAdamDefaultConfiguration(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{
			Device:    dev,
			GradTrack: true,
		}

		optimizer, err := optimizers.NewAdam(nil)
		if err != nil {
			t.Fatal(err)
		}

		x, err := tensor.Full(nil, 1., conf)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		x.ResetGradContext(true)

		y := x.Scale(4.).Scale(5.).Scale(5.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x)
		if err != nil {
			t.Fatal(err)
		}

		act := x

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.999-1e-10 < val && val < 0.999+1e-10) {
			t.Fatalf("expected scalar tensors value to be (0.999): got (%f)", val)
		}

		/* ------------------------------ */

		x.ResetGradContext(true)

		y = x.Scale(4.).Scale(5.).Scale(5.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x)
		if err != nil {
			t.Fatal(err)
		}

		act = x

		val, err = act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.998-1e-10 < val && val < 0.998+1e-10) {
			t.Fatalf("expected scalar tensors value to be (0.998): got (%f)", val)
		}

		/* ------------------------------ */

	})
}

func TestAdamWithWeightDecay(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{
			Device:    dev,
			GradTrack: true,
		}

		optimizer, err := optimizers.NewAdam(&optimizers.AdamConfig{
			LearningRate: 2.,
			WeightDecay:  2.,
			Beta1:        0.5,
			Beta2:        0.75,
			Eps:          100.,
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err := tensor.Full(nil, 2., conf)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		x.ResetGradContext(true)

		y := x.Scale(4.).Scale(5.).Scale(5.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		err = optimizer.Update(&x)
		if err != nil {
			t.Fatal(err)
		}

		act := x

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.9803-1e-4 < val && val < 0.9803+1e-4) {
			t.Fatalf("expected scalar tensors value to be (0.9803): got (%f)", val)
		}

		/* ------------------------------ */

	})
}

func TestValidationAdam(t *testing.T) {
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

		_, err := optimizers.NewAdam(&optimizers.AdamConfig{})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'LearningRate'")
		} else if err.Error() != "Adam config data validation failed: expected 'LearningRate' to be positive: got (0.000000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = optimizers.NewAdam(&optimizers.AdamConfig{
			LearningRate: -1,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'LearningRate'")
		} else if err.Error() != "Adam config data validation failed: expected 'LearningRate' to be positive: got (-1.000000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = optimizers.NewAdam(&optimizers.AdamConfig{
			LearningRate: 1,
			WeightDecay:  -1,
		})
		if err == nil {
			t.Fatalf("expected error because of negative 'WeightDecay'")
		} else if err.Error() != "Adam config data validation failed: expected 'WeightDecay' not to be negative: got (-1.000000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = optimizers.NewAdam(&optimizers.AdamConfig{
			LearningRate: 10,
			WeightDecay:  0,
			Beta1:        -0.5,
		})
		if err == nil {
			t.Fatalf("expected error because of negative 'Beta1'")
		} else if err.Error() != "Adam config data validation failed: expected 'Beta1' not to be negative: got (-0.500000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = optimizers.NewAdam(&optimizers.AdamConfig{
			LearningRate: 10,
			WeightDecay:  1,
			Beta1:        0,
			Beta2:        -0.1,
		})
		if err == nil {
			t.Fatalf("expected error because of negative 'Beta2'")
		} else if err.Error() != "Adam config data validation failed: expected 'Beta2' not to be negative: got (-0.100000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = optimizers.NewAdam(&optimizers.AdamConfig{
			LearningRate: 100,
			WeightDecay:  10,
			Beta1:        1,
			Beta2:        0,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'Eps'")
		} else if err.Error() != "Adam config data validation failed: expected 'Eps' to be positive: got (0.000000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = optimizers.NewAdam(&optimizers.AdamConfig{
			LearningRate: 100,
			WeightDecay:  10,
			Beta1:        1,
			Beta2:        0.5,
			Eps:          -1,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'Eps'")
		} else if err.Error() != "Adam config data validation failed: expected 'Eps' to be positive: got (-1.000000)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		optimizer, err := optimizers.NewAdam(nil)
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
		} else if err.Error() != "Adam input data validation failed: expected tensor's gradient not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		err = optimizer.Update(&wt)
		if err == nil {
			t.Fatalf("expected error because of nil tensor gradient")
		} else if err.Error() != "Adam input data validation failed: expected tensor's gradient not to be nil" {
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
		} else if err.Error() != "Adam input data validation failed: expected tensor's gradient not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
