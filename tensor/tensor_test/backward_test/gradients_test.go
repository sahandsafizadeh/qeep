package backward_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func TestConcat(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x1, err := tinit.Zeros(conf, 5, 1, 3)
		if err != nil {
			t.Fatal(err)
		}

		x2, err := tinit.Zeros(conf, 5, 5, 3)
		if err != nil {
			t.Fatal(err)
		}

		x3, err := tinit.Zeros(conf, 5, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		y, err := tinit.Concat([]tensor.Tensor{x1, x2, x3}, 1)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x1.Gradient()

		exp, err := tinit.Ones(conf, 5, 1, 3)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		act = x2.Gradient()

		exp, err = tinit.Ones(conf, 5, 5, 3)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		act = x3.Gradient()

		exp, err = tinit.Ones(conf, 5, 2, 3)
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

func TestSlice(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.TensorOf(conf, [][]float64{
			{0., 1., 2., 3., 4.},
			{5., 6., 7., 8., 9.},
			{4., 3., 2., 1., 0.},
			{9., 8., 7., 6., 5.},
		})
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.Slice([]tensor.Range{{From: 1, To: 4}, {From: 1, To: 4}})
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.TensorOf(conf, [][]float64{
			{0., 0., 0., 0., 0.},
			{0., 1., 1., 1., 0.},
			{0., 1., 1., 1., 0.},
			{0., 1., 1., 1., 0.},
		})
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

func TestPatch(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.TensorOf(conf, [][]float64{
			{0., 1., 2., 3., 4.},
			{5., 6., 7., 8., 9.},
			{4., 3., 2., 1., 0.},
			{9., 8., 7., 6., 5.},
		})
		if err != nil {
			t.Fatal(err)
		}

		p, err := tinit.TensorOf(conf, [][]float64{
			{-1., -2., -3.},
			{-4., -5., -6.},
			{-7., -8., -9.},
		})
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.Patch([]tensor.Range{{From: 1, To: 4}, {From: 1, To: 4}}, p)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.TensorOf(conf, [][]float64{
			{1., 1., 1., 1., 1.},
			{1., 0., 0., 0., 1.},
			{1., 0., 0., 0., 1.},
			{1., 0., 0., 0., 1.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		act = p.Gradient()

		exp, err = tinit.TensorOf(conf, [][]float64{
			{1., 1., 1.},
			{1., 1., 1.},
			{1., 1., 1.},
		})
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

func TestTranspose(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Zeros(conf, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.Ones(conf, 3, 4)
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

func TestReshape(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Zeros(conf, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.Reshape(6, 2)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.Ones(conf, 3, 4)
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

func TestUnsqueeze(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Zeros(conf, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.UnSqueeze(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.Ones(conf, 3, 4)
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

func TestSqueeze(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Zeros(conf, 3, 1, 4)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.Squeeze(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.Ones(conf, 3, 1, 4)
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

func TestFlatten(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Zeros(conf, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.Flatten(0)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.Ones(conf, 3, 4)
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

func TestBroadcast(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Zeros(conf, 3, 1, 4)
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.Broadcast(6, 5, 3, 2, 4)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.Ones(conf, 3, 1, 4)
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

func TestSum(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.TensorOf(conf, [][][]float64{
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
		})
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.SumAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.TensorOf(conf, [][][]float64{
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
		})
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

func TestMax(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.TensorOf(conf, [][][]float64{
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
		})
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.MaxAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.TensorOf(conf, [][][]float64{
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
		})
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

func TestMin(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.TensorOf(conf, [][][]float64{
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
		})
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.MinAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.TensorOf(conf, [][][]float64{
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
		})
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

func TestAvg(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.TensorOf(conf, [][][]float64{
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
		})
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.AvgAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.TensorOf(conf, [][][]float64{
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
		})
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

func TestVar(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.TensorOf(conf, [][][]float64{
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
		})
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.VarAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.TensorOf(conf, [][][]float64{
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
		})
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

func TestStd(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.TensorOf(conf, [][][]float64{
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
		})
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.StdAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.TensorOf(conf, [][][]float64{
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
		})
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

func TestMean(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.TensorOf(conf, [][][]float64{
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
		})
		if err != nil {
			t.Fatal(err)
		}

		y, err := x.MeanAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.TensorOf(conf, [][][]float64{
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
		})
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

func TestScale(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Full(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Scale(3.)

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.Full(conf, 3.)
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

func TestPow(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Full(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Pow(3.)

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.Full(conf, 12.)
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

func TestExp(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Full(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Exp()

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.Full(conf, math.E)
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

func TestLog(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Full(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Log()

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		exp, err := tinit.Full(conf, 0.5)
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

func TestSin(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Full(conf, math.Pi/3)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Sin()

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(0.5-1e-10 < val && val < 0.5+1e-10) {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestCos(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Full(conf, math.Pi/6)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Cos()

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(-0.5-1e-10 < val && val < -0.5+1e-10) {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestTan(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Full(conf, 0.)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Tan()

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(1.-1e-10 < val && val < 1.+1e-10) {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestSinh(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Full(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Sinh()

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()
		c := (math.E*math.E + 1) / (2 * math.E)

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(c-1e-10 < val && val < c+1e-10) {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestCosh(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Full(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Cosh()

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()
		c := (math.E*math.E - 1) / (2 * math.E)

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(c-1e-10 < val && val < c+1e-10) {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestTanh(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		x, err := tinit.Full(conf, 0.)
		if err != nil {
			t.Fatal(err)
		}

		y := x.Tanh()

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := x.Gradient()

		val, err := act.At()
		if err != nil {
			t.Fatal(err)
		} else if !(1.-1e-10 < val && val < 1.+1e-10) {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

	})
}

func TestAdd(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		a, err := tinit.Full(conf, 3.)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tinit.Full(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		y, err := a.Add(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := a.Gradient()

		exp, err := tinit.Full(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		act = b.Gradient()

		exp, err = tinit.Full(conf, 1.)
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

func TestSub(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		a, err := tinit.Full(conf, 3.)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tinit.Full(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		y, err := a.Sub(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := a.Gradient()

		exp, err := tinit.Full(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		act = b.Gradient()

		exp, err = tinit.Full(conf, -1.)
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

func TestMul(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		a, err := tinit.Full(conf, 3.)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tinit.Full(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		y, err := a.Mul(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := a.Gradient()

		exp, err := tinit.Full(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		act = b.Gradient()

		exp, err = tinit.Full(conf, 3.)
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

func TestDiv(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		a, err := tinit.Full(conf, 3.)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tinit.Full(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		y, err := a.Div(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := a.Gradient()

		exp, err := tinit.Full(conf, 0.5)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		act = b.Gradient()

		exp, err = tinit.Full(conf, -0.75)
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

func TestDot(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		a, err := tinit.Full(conf, 2., 2)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tinit.Full(conf, 3., 2)
		if err != nil {
			t.Fatal(err)
		}

		y, err := a.Dot(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := a.Gradient()

		exp, err := tinit.Full(conf, 3., 2)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		act = b.Gradient()

		exp, err = tinit.Full(conf, 2., 2)
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

func TestMatmul(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		a, err := tinit.Full(conf, 2., 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tinit.Full(conf, 3., 3, 2)
		if err != nil {
			t.Fatal(err)
		}

		y, err := a.MatMul(b)
		if err != nil {
			t.Fatal(err)
		}

		err = tinit.BackProp(y)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		act := a.Gradient()

		exp, err := tinit.Full(conf, 6., 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		act = b.Gradient()

		exp, err = tinit.Full(conf, 4., 3, 2)
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
