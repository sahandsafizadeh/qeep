package backward_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestConcat(t *testing.T) {
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

		x1, err := tensor.RandN([]int{5, 1, 3}, 0., 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		x2, err := tensor.RandN([]int{5, 5, 3}, 0., 1., confT)
		if err != nil {
			t.Fatal(err)
		}

		x3, err := tensor.RandU([]int{5, 2, 3}, -1., 1., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := tensor.Concat([]tensor.Tensor{x1, x2, x3}, 1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x1.Gradient()

		exp, err := tensor.Ones([]int{5, 1, 3}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* --------------- */

		act = x2.Gradient()

		exp, err = tensor.Ones([]int{5, 5, 3}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* --------------- */

		act = x3.Gradient()

		exp, err = tensor.Ones([]int{5, 2, 3}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x1, err = tensor.Zeros([]int{1}, confT)
		if err != nil {
			t.Fatal(err)
		}

		x2, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = tensor.Concat([]tensor.Tensor{x1, x2}, 0)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		x1, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		x2, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = tensor.Concat([]tensor.Tensor{x1, x2}, 0)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestSlice(t *testing.T) {
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

		x, err := tensor.Of([][]float64{
			{0., 1., 2., 3., 4.},
			{5., 6., 7., 8., 9.},
			{4., 3., 2., 1., 0.},
			{9., 8., 7., 6., 5.},
		}, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.Slice([]tensor.Range{{From: 1, To: 4}, {From: 1, To: 4}})
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Of([][]float64{
			{0., 0., 0., 0., 0.},
			{0., 1., 1., 1., 0.},
			{0., 1., 1., 1., 0.},
			{0., 1., 1., 1., 0.},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.Slice(nil)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestPatch(t *testing.T) {
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

		x, err := tensor.Of([][]float64{
			{0., 1., 2., 3., 4.},
			{5., 6., 7., 8., 9.},
			{4., 3., 2., 1., 0.},
			{9., 8., 7., 6., 5.},
		}, confT)
		if err != nil {
			t.Fatal(err)
		}

		p, err := tensor.Of([][]float64{
			{-1., -2., -3.},
			{-4., -5., -6.},
			{-7., -8., -9.},
		}, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.Patch([]tensor.Range{{From: 1, To: 4}, {From: 1, To: 4}}, p)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Of([][]float64{
			{1., 1., 1., 1., 1.},
			{1., 0., 0., 0., 1.},
			{1., 0., 0., 0., 1.},
			{1., 0., 0., 0., 1.},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* --------------- */

		act = p.Gradient()

		exp, err = tensor.Of([][]float64{
			{1., 1., 1.},
			{1., 1., 1.},
			{1., 1., 1.},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		p, err = tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.Patch(nil, p)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		p, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.Patch(nil, p)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		p, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.Patch(nil, p)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestTranspose(t *testing.T) {
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

		x, err := tensor.RandU([]int{3, 4}, -1., 1., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Ones([]int{3, 4}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros([]int{1, 1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestReshape(t *testing.T) {
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

		x, err := tensor.RandN([]int{3, 4}, 0., 1., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.Reshape([]int{6, 2})
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Ones([]int{3, 4}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.Reshape(nil)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestUnsqueeze(t *testing.T) {
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

		x, err := tensor.RandN([]int{3, 4}, 0., 1., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.UnSqueeze(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Ones([]int{3, 4}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.UnSqueeze(0)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestSqueeze(t *testing.T) {
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

		x, err := tensor.RandN([]int{3, 1, 4}, 0., 1., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.Squeeze(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Ones([]int{3, 1, 4}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.Squeeze(0)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestFlatten(t *testing.T) {
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

		x, err := tensor.RandN([]int{3, 4}, 0., 1., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.Flatten(0)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Ones([]int{3, 4}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.Flatten(0)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestBroadcast(t *testing.T) {
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

		x, err := tensor.RandU([]int{3, 1, 4}, 0., 1., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.Broadcast([]int{6, 5, 3, 3, 4})
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Ones([]int{3, 1, 4}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.Broadcast(nil)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestSum(t *testing.T) {
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

		x, err := tensor.Of([][][]float64{
			{
				{1., 1., 1., 1.},
				{2., 2., 2., 2.},
				{3., 3., 3., 3.},
				{4., 4., 4., 4.},
			},
			{
				{1., 1., 1., 1.},
				{2., 2., 2., 2.},
				{3., 3., 3., 3.},
				{4., 4., 4., 4.},
			},
		}, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.SumAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Of([][][]float64{
			{
				{1., 1., 1., 1.},
				{1., 1., 1., 1.},
				{1., 1., 1., 1.},
				{1., 1., 1., 1.},
			},
			{
				{1., 1., 1., 1.},
				{1., 1., 1., 1.},
				{1., 1., 1., 1.},
				{1., 1., 1., 1.},
			},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.SumAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestMax(t *testing.T) {
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

		x, err := tensor.Of([][][]float64{
			{
				{1., 1., 1., 1.},
				{2., 2., 2., 2.},
				{3., 3., 3., 3.},
				{4., 4., 4., 4.},
			},
			{
				{1., 1., 1., 1.},
				{2., 2., 2., 2.},
				{3., 3., 3., 3.},
				{4., 4., 4., 4.},
			},
		}, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.MaxAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Of([][][]float64{
			{
				{0., 0., 0., 0.},
				{0., 0., 0., 0.},
				{0., 0., 0., 0.},
				{1., 1., 1., 1.},
			},
			{
				{0., 0., 0., 0.},
				{0., 0., 0., 0.},
				{0., 0., 0., 0.},
				{1., 1., 1., 1.},
			},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.MaxAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestMin(t *testing.T) {
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

		x, err := tensor.Of([][][]float64{
			{
				{1., 1., 1., 1.},
				{2., 2., 2., 2.},
				{3., 3., 3., 3.},
				{4., 4., 4., 4.},
			},
			{
				{1., 1., 1., 1.},
				{2., 2., 2., 2.},
				{3., 3., 3., 3.},
				{4., 4., 4., 4.},
			},
		}, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.MinAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Of([][][]float64{
			{
				{1., 1., 1., 1.},
				{0., 0., 0., 0.},
				{0., 0., 0., 0.},
				{0., 0., 0., 0.},
			},
			{
				{1., 1., 1., 1.},
				{0., 0., 0., 0.},
				{0., 0., 0., 0.},
				{0., 0., 0., 0.},
			},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.MinAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestAvg(t *testing.T) {
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

		x, err := tensor.Of([][][]float64{
			{
				{1., 1., 1., 1.},
				{2., 2., 2., 2.},
				{3., 3., 3., 3.},
				{4., 4., 4., 4.},
			},
			{
				{1., 1., 1., 1.},
				{2., 2., 2., 2.},
				{3., 3., 3., 3.},
				{4., 4., 4., 4.},
			},
		}, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.AvgAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Of([][][]float64{
			{
				{0.25, 0.25, 0.25, 0.25},
				{0.25, 0.25, 0.25, 0.25},
				{0.25, 0.25, 0.25, 0.25},
				{0.25, 0.25, 0.25, 0.25},
			},
			{
				{0.25, 0.25, 0.25, 0.25},
				{0.25, 0.25, 0.25, 0.25},
				{0.25, 0.25, 0.25, 0.25},
				{0.25, 0.25, 0.25, 0.25},
			},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.AvgAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestVar(t *testing.T) {
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

		x, err := tensor.Ones([]int{5, 1, 3}, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.VarAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Zeros([]int{5, 1, 3}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Of([][][]float64{
			{
				{1., 1., 1., 1.},
				{2., 2., 2., 2.},
				{3., 3., 3., 3.},
			},
			{
				{1., 1., 1., 1.},
				{2., 2., 2., 2.},
				{3., 3., 3., 3.},
			},
		}, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.VarAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act = x.Gradient()

		exp, err = tensor.Of([][][]float64{
			{
				{-1., -1., -1., -1.},
				{0., 0., 0., 0.},
				{1., 1., 1., 1.},
			},
			{
				{-1., -1., -1., -1.},
				{0., 0., 0., 0.},
				{1., 1., 1., 1.},
			},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.VarAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestStd(t *testing.T) {
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

		x, err := tensor.Ones([]int{5, 1, 3}, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.StdAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Zeros([]int{5, 1, 3}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Of([][][]float64{
			{
				{1., 1., 1., 1.},
				{2., 2., 2., 2.},
				{3., 3., 3., 3.},
			},
			{
				{1., 1., 1., 1.},
				{2., 2., 2., 2.},
				{3., 3., 3., 3.},
			},
		}, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.StdAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act = x.Gradient()

		exp, err = tensor.Of([][][]float64{
			{
				{-0.5, -0.5, -0.5, -0.5},
				{0., 0., 0., 0.},
				{0.5, 0.5, 0.5, 0.5},
			},
			{
				{-0.5, -0.5, -0.5, -0.5},
				{0., 0., 0., 0.},
				{0.5, 0.5, 0.5, 0.5},
			},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.StdAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestMean(t *testing.T) {
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

		x, err := tensor.Of([][][]float64{
			{
				{1., 1., 1., 1.},
				{2., 2., 2., 2.},
				{3., 3., 3., 3.},
				{4., 4., 4., 4.},
			},
			{
				{1., 1., 1., 1.},
				{2., 2., 2., 2.},
				{3., 3., 3., 3.},
				{4., 4., 4., 4.},
			},
		}, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.MeanAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Of([][][]float64{
			{
				{0.25, 0.25, 0.25, 0.25},
				{0.25, 0.25, 0.25, 0.25},
				{0.25, 0.25, 0.25, 0.25},
				{0.25, 0.25, 0.25, 0.25},
			},
			{
				{0.25, 0.25, 0.25, 0.25},
				{0.25, 0.25, 0.25, 0.25},
				{0.25, 0.25, 0.25, 0.25},
				{0.25, 0.25, 0.25, 0.25},
			},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = x.MeanAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestScale(t *testing.T) {
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

		x, err := tensor.Full(nil, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Scale(3.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Full(nil, 3., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y = x.Scale(0.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestPow(t *testing.T) {
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

		x, err := tensor.Full(nil, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Pow(3.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Full(nil, 12., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y = x.Pow(0.)

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestExp(t *testing.T) {
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

		x, err := tensor.Full(nil, 1., confT)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Exp()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Full(nil, math.E, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y = x.Exp()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestLog(t *testing.T) {
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

		x, err := tensor.Full(nil, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Log()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		exp, err := tensor.Full(nil, 0.5, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y = x.Log()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestSin(t *testing.T) {
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

		x, err := tensor.Full(nil, math.Pi/3, confT)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Sin()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.5-1e-10 < val && val < 0.5+1e-10) {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y = x.Sin()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestCos(t *testing.T) {
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

		x, err := tensor.Full(nil, math.Pi/6, confT)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Cos()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(-0.5-1e-10 < val && val < -0.5+1e-10) {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y = x.Cos()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestTan(t *testing.T) {
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

		x, err := tensor.Full(nil, 0., confT)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Tan()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(1.-1e-10 < val && val < 1.+1e-10) {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y = x.Tan()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestSinh(t *testing.T) {
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

		x, err := tensor.Full(nil, 1., confT)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Sinh()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()
		c := (math.E*math.E + 1) / (2 * math.E)

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(c-1e-10 < val && val < c+1e-10) {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y = x.Sinh()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestCosh(t *testing.T) {
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

		x, err := tensor.Full(nil, 1., confT)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Cosh()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()
		c := (math.E*math.E - 1) / (2 * math.E)

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(c-1e-10 < val && val < c+1e-10) {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y = x.Cosh()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestTanh(t *testing.T) {
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

		x, err := tensor.Full(nil, 0., confT)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Tanh()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := x.Gradient()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(1.-1e-10 < val && val < 1.+1e-10) {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		x, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y = x.Tanh()

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestElMax(t *testing.T) {
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

		a, err := tensor.Full(nil, 3., confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tensor.Full(nil, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := a.ElMax(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := a.Gradient()

		exp, err := tensor.Full(nil, 1., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* --------------- */

		act = b.Gradient()

		exp, err = tensor.Full(nil, 0., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Full(nil, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Full(nil, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.ElMax(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act = a.Gradient()

		exp, err = tensor.Full(nil, 0.5, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* --------------- */

		act = b.Gradient()

		exp, err = tensor.Full(nil, 0.5, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.ElMax(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.ElMax(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.ElMax(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestElMin(t *testing.T) {
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

		a, err := tensor.Full(nil, 3., confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tensor.Full(nil, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := a.ElMin(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := a.Gradient()

		exp, err := tensor.Full(nil, 0., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* --------------- */

		act = b.Gradient()

		exp, err = tensor.Full(nil, 1., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Full(nil, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Full(nil, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.ElMin(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act = a.Gradient()

		exp, err = tensor.Full(nil, 0.5, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* --------------- */

		act = b.Gradient()

		exp, err = tensor.Full(nil, 0.5, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.ElMin(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.ElMin(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.ElMin(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestAdd(t *testing.T) {
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

		a, err := tensor.Full(nil, 3., confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tensor.Full(nil, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := a.Add(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := a.Gradient()

		exp, err := tensor.Full(nil, 1., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* --------------- */

		act = b.Gradient()

		exp, err = tensor.Full(nil, 1., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Add(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Add(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Add(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestSub(t *testing.T) {
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

		a, err := tensor.Full(nil, 3., confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tensor.Full(nil, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := a.Sub(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := a.Gradient()

		exp, err := tensor.Full(nil, 1., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* --------------- */

		act = b.Gradient()

		exp, err = tensor.Full(nil, -1., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Sub(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Sub(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Sub(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestMul(t *testing.T) {
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

		a, err := tensor.Full(nil, 3., confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tensor.Full(nil, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := a.Mul(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := a.Gradient()

		exp, err := tensor.Full(nil, 2., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* --------------- */

		act = b.Gradient()

		exp, err = tensor.Full(nil, 3., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Mul(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Mul(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Mul(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestDiv(t *testing.T) {
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

		a, err := tensor.Full(nil, 3., confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tensor.Full(nil, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := a.Div(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := a.Gradient()

		exp, err := tensor.Full(nil, 0.5, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* --------------- */

		act = b.Gradient()

		exp, err = tensor.Full(nil, -0.75, confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Div(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Div(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros(nil, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Div(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestDot(t *testing.T) {
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

		a, err := tensor.Full([]int{2}, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tensor.Full([]int{2}, 3., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := a.Dot(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := a.Gradient()

		exp, err := tensor.Full([]int{2}, 3., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* --------------- */

		act = b.Gradient()

		exp, err = tensor.Full([]int{2}, 2., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros([]int{1}, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Dot(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros([]int{1}, confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Dot(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Zeros([]int{1}, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.Dot(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}

func TestMatmul(t *testing.T) {
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

		a, err := tensor.Full([]int{2, 3}, 2., confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tensor.Full([]int{3, 2}, 3., confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err := a.MatMul(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := a.Gradient()

		exp, err := tensor.Full([]int{2, 3}, 6., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* --------------- */

		act = b.Gradient()

		exp, err = tensor.Full([]int{3, 2}, 4., confU)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		} else if act.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Eye(1, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Eye(1, confT)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.MatMul(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Eye(1, confT)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Eye(1, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.MatMul(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() == nil {
			t.Fatalf("expected gradient not to be nil")
		}

		/* ------------------------------ */

		a, err = tensor.Eye(1, confU)
		if err != nil {
			t.Fatal(err)
		}

		b, err = tensor.Eye(1, confU)
		if err != nil {
			t.Fatal(err)
		}

		y, err = a.MatMul(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tensor.BackPropagate(y)
		if err != nil {
			t.Fatal(err)
		}

		if y.Gradient() != nil {
			t.Fatalf("expected gradient to be nil")
		}

		/* ------------------------------ */

	})
}
