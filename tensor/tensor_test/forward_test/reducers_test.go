package forward_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func TestSum(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 9.)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Sum(); int(val) != 9 {
			t.Fatalf("expected (9) as the sum value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{6., 4.})
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Sum(); int(val) != 10 {
			t.Fatalf("expected (10) as the sum value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][][]float64{
			{{9., -1., 8., 6.}},
			{{-5., 4., 1., 0.}},
			{{2., 8., 7., -3.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Sum(); int(val) != 36 {
			t.Fatalf("expected (30) as the sum value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.SumAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.SumAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, 3.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 3, 4, 5)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.SumAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Full(conf, 3., 4, 5)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 3, 4, 5)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.SumAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Full(conf, 4., 3, 5)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 3, 4, 5)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.SumAlong(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Full(conf, 5., 3, 4)
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

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 9.)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Max(); int(val) != 9 {
			t.Fatalf("expected (9) as the max value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{6., 4.})
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Max(); int(val) != 6 {
			t.Fatalf("expected (6) as the max value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][][]float64{
			{{9., -1., 8., 6.}},
			{{-5., 4., 1., 0.}},
			{{2., 8., 7., -3.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Max(); int(val) != 9 {
			t.Fatalf("expected (9) as the max value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.MaxAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MaxAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][][]float64{
			{
				{1., 2., -5.},
				{0., -1., 3.},
				{7., -7., 7.},
			},
			{
				{8., -1., 0.},
				{5., 4., -3.},
				{1., -3., 5.},
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MaxAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{8., 2., 0.},
			{5., 4., 3.},
			{7., -3., 7.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.MaxAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{7., 2., 7.},
			{8., 4., 5.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.MaxAlong(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{2., 3., 7.},
			{8., 5., 5.},
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

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 9.)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Min(); int(val) != 9 {
			t.Fatalf("expected (9) as the min value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{6., 4.})
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Min(); int(val) != 4 {
			t.Fatalf("expected (4) as the min value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][][]float64{
			{{9., -1., 8., 6.}},
			{{-5., 4., 1., 0.}},
			{{2., 8., 7., -3.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Min(); int(val) != -5 {
			t.Fatalf("expected (-5) as the max value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.MinAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MinAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][][]float64{
			{
				{1., 2., -5.},
				{0., -1., 3.},
				{7., -7., 7.},
			},
			{
				{8., -1., 0.},
				{5., 4., -3.},
				{1., -3., 5.},
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MinAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{1., -1., -5.},
			{0., -1., -3.},
			{1., -7., 5.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.MinAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{0., -7., -5.},
			{1., -3., -3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.MinAlong(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{-5., -1., -7.},
			{-1., -3., -3.},
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

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 9.)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Avg(); int(val) != 9 {
			t.Fatalf("expected (9) as the avg value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{6., 4.})
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Avg(); int(val) != 5 {
			t.Fatalf("expected (5) as the avg value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][][]float64{
			{{9., -1., 8., 6.}},
			{{-5., 4., 1., 0.}},
			{{2., 8., 7., -3.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Avg(); int(val) != 3 {
			t.Fatalf("expected (3) as the avg value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.AvgAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Full(conf, 2., 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.AvgAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Full(conf, 3., 3, 4, 5)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.AvgAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Full(conf, 3., 4, 5)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Full(conf, 4., 3, 4, 5)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.AvgAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Full(conf, 4., 3, 5)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Full(conf, 5., 3, 4, 5)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.AvgAlong(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Full(conf, 5., 3, 4)
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

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 9.)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Var(); int(val) != 0 {
			t.Fatalf("expected (0) as the var value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{-math.Sqrt2, math.Sqrt2})
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Var(); int(val) != 4 {
			t.Fatalf("expected (4) as the var value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][][]float64{{{-2.}, {0.}, {2.}}})
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Var(); int(val) != 4 {
			t.Fatalf("expected (4) as the var value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.VarAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 0.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Full(conf, 5., 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.VarAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, 0.)
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

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 9.)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Std(); int(val) != 0 {
			t.Fatalf("expected (0) as the std value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{-math.Sqrt2, math.Sqrt2})
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Std(); int(val) != 2 {
			t.Fatalf("expected (2) as the std value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][][]float64{{{-2.}, {0.}, {2.}}})
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Std(); int(val) != 2 {
			t.Fatalf("expected (2) as the std value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.StdAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 0.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Full(conf, 5., 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.StdAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, 0.)
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

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 9.)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Mean(); int(val) != 9 {
			t.Fatalf("expected (9) as the mean value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{6., 4.})
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Mean(); int(val) != 5 {
			t.Fatalf("expected (5) as the mean value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][][]float64{
			{{9., -1., 8., 6.}},
			{{-5., 4., 1., 0.}},
			{{2., 8., 7., -3.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Mean(); int(val) != 3 {
			t.Fatalf("expected (3) as the mean value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.Ones(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.MeanAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 1.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Full(conf, 2., 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MeanAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Full(conf, 3., 3, 4, 5)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MeanAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Full(conf, 3., 4, 5)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Full(conf, 4., 3, 4, 5)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MeanAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Full(conf, 4., 3, 5)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Full(conf, 5., 3, 4, 5)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MeanAlong(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Full(conf, 5., 3, 4)
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

func TestValidationReducers(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.SumAlong(-1)
		if err == nil {
			t.Fatalf("expected error because of negative dimension")
		}

		_, err = ten.SumAlong(0)
		if err == nil {
			t.Fatalf("expected error because of reduced dimension (0) being out of range")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.SumAlong(1)
		if err == nil {
			t.Fatalf("expected error because of reduced dimension (1) being out of range")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 3, 1)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.SumAlong(2)
		if err == nil {
			t.Fatalf("expected error because of reduced dimension (2) being out of range")
		}

		/* ------------------------------ */

	})
}
