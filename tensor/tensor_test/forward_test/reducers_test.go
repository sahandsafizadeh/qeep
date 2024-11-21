package forward_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestSum(t *testing.T) {
	runTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tensor.TensorOf(9., conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Sum(); int(val) != 9 {
			t.Fatalf("expected (9) as the sum value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([]float64{6., 4.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Sum(); int(val) != 10 {
			t.Fatalf("expected (10) as the sum value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([][][]float64{
			{{9., -1., 8., 6.}},
			{{-5., 4., 1., 0.}},
			{{2., 8., 7., -3.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Sum(); int(val) != 36 {
			t.Fatalf("expected (30) as the sum value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.Ones([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.SumAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Ones([]int{3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.SumAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.TensorOf(3., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Ones([]int{3, 4, 5}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.SumAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Full([]int{4, 5}, 3., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Ones([]int{3, 4, 5}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.SumAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Full([]int{3, 5}, 4., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Ones([]int{3, 4, 5}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.SumAlong(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Full([]int{3, 4}, 5., conf)
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
	runTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tensor.TensorOf(9., conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Max(); int(val) != 9 {
			t.Fatalf("expected (9) as the max value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([]float64{6., 4.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Max(); int(val) != 6 {
			t.Fatalf("expected (6) as the max value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([][][]float64{
			{{9., -1., 8., 6.}},
			{{-5., 4., 1., 0.}},
			{{2., 8., 7., -3.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Max(); int(val) != 9 {
			t.Fatalf("expected (9) as the max value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.Ones([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.MaxAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Ones([]int{3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MaxAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([][][]float64{
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
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MaxAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.TensorOf([][]float64{
			{8., 2., 0.},
			{5., 4., 3.},
			{7., -3., 7.},
		}, conf)
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

		exp, err = tensor.TensorOf([][]float64{
			{7., 2., 7.},
			{8., 4., 5.},
		}, conf)
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

		exp, err = tensor.TensorOf([][]float64{
			{2., 3., 7.},
			{8., 5., 5.},
		}, conf)
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
	runTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tensor.TensorOf(9., conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Min(); int(val) != 9 {
			t.Fatalf("expected (9) as the min value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([]float64{6., 4.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Min(); int(val) != 4 {
			t.Fatalf("expected (4) as the min value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([][][]float64{
			{{9., -1., 8., 6.}},
			{{-5., 4., 1., 0.}},
			{{2., 8., 7., -3.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Min(); int(val) != -5 {
			t.Fatalf("expected (-5) as the max value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.Ones([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.MinAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Ones([]int{3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MinAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([][][]float64{
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
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MinAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.TensorOf([][]float64{
			{1., -1., -5.},
			{0., -1., -3.},
			{1., -7., 5.},
		}, conf)
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

		exp, err = tensor.TensorOf([][]float64{
			{0., -7., -5.},
			{1., -3., -3.},
		}, conf)
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

		exp, err = tensor.TensorOf([][]float64{
			{-5., -1., -7.},
			{-1., -3., -3.},
		}, conf)
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
	runTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tensor.TensorOf(9., conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Avg(); int(val) != 9 {
			t.Fatalf("expected (9) as the avg value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([]float64{6., 4.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Avg(); int(val) != 5 {
			t.Fatalf("expected (5) as the avg value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([][][]float64{
			{{9., -1., 8., 6.}},
			{{-5., 4., 1., 0.}},
			{{2., 8., 7., -3.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Avg(); int(val) != 3 {
			t.Fatalf("expected (3) as the avg value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.Ones([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.AvgAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Full([]int{3}, 2., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.AvgAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.TensorOf(2., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Full([]int{3, 4, 5}, 3., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.AvgAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Full([]int{4, 5}, 3., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Full([]int{3, 4, 5}, 4., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.AvgAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Full([]int{3, 5}, 4., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Full([]int{3, 4, 5}, 5., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.AvgAlong(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Full([]int{3, 4}, 5., conf)
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
	runTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tensor.TensorOf(9., conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Var(); int(val) != 0 {
			t.Fatalf("expected (0) as the var value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([]float64{-math.Sqrt2, math.Sqrt2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Var(); int(val) != 4 {
			t.Fatalf("expected (4) as the var value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([][][]float64{{{-2.}, {0.}, {2.}}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Var(); int(val) != 4 {
			t.Fatalf("expected (4) as the var value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.Ones([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.VarAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.TensorOf(0., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Full([]int{3}, 5., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.VarAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.TensorOf(0., conf)
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
	runTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tensor.TensorOf(9., conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Std(); int(val) != 0 {
			t.Fatalf("expected (0) as the std value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([]float64{-math.Sqrt2, math.Sqrt2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Std(); int(val) != 2 {
			t.Fatalf("expected (2) as the std value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([][][]float64{{{-2.}, {0.}, {2.}}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Std(); int(val) != 2 {
			t.Fatalf("expected (2) as the std value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.Ones([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.StdAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.TensorOf(0., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Full([]int{3}, 5., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.StdAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.TensorOf(0., conf)
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
	runTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tensor.TensorOf(9., conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Mean(); int(val) != 9 {
			t.Fatalf("expected (9) as the mean value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([]float64{6., 4.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Mean(); int(val) != 5 {
			t.Fatalf("expected (5) as the mean value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.TensorOf([][][]float64{
			{{9., -1., 8., 6.}},
			{{-5., 4., 1., 0.}},
			{{2., 8., 7., -3.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val := ten.Mean(); int(val) != 3 {
			t.Fatalf("expected (3) as the mean value of tensor, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.Ones([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.MeanAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.TensorOf(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Full([]int{3}, 2., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MeanAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.TensorOf(2., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Full([]int{3, 4, 5}, 3., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MeanAlong(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Full([]int{4, 5}, 3., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Full([]int{3, 4, 5}, 4., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MeanAlong(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Full([]int{3, 5}, 4., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Full([]int{3, 4, 5}, 5., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.MeanAlong(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Full([]int{3, 4}, 5., conf)
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
	runTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tensor.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.SumAlong(-1)
		if err == nil {
			t.Fatalf("expected error because of reduced dimension (-1) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.SumAlong(0)
		if err == nil {
			t.Fatalf("expected error because of reduced dimension (0) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = tensor.Zeros([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.SumAlong(1)
		if err == nil {
			t.Fatalf("expected error because of reduced dimension (1) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = tensor.Zeros([]int{3, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.SumAlong(2)
		if err == nil {
			t.Fatalf("expected error because of reduced dimension (2) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.MaxAlong(2)
		if err == nil {
			t.Fatalf("expected error because of reduced dimension (2) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.MinAlong(2)
		if err == nil {
			t.Fatalf("expected error because of reduced dimension (2) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.AvgAlong(2)
		if err == nil {
			t.Fatalf("expected error because of reduced dimension (2) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.VarAlong(2)
		if err == nil {
			t.Fatalf("expected error because of reduced dimension (2) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.StdAlong(2)
		if err == nil {
			t.Fatalf("expected error because of reduced dimension (2) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.MeanAlong(2)
		if err == nil {
			t.Fatalf("expected error because of reduced dimension (2) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
