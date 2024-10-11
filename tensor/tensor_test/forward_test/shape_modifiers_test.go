package forward_test

import (
	"testing"

	qti "github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func TestTranspose(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.TensorOf([][]float64{{1.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf([][]float64{{1.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf([][]float64{{1., 0., 2.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{{1.}, {0.}, {2.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf([][]float64{{-2.}, {0.}, {-1.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{{-2., 0., -1.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf([][]float64{
			{0., 1., 2., 3.},
			{0., 1., 2., 3.},
			{0., 1., 2., 3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{
			{0., 0., 0.},
			{1., 1., 1.},
			{2., 2., 2.},
			{3., 3., 3.},
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

		ten, err = qti.TensorOf([][][]float64{
			{
				{1., 2.},
				{3., 4.},
			},
			{
				{1., 2.},
				{3., 4.},
			},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][][]float64{
			{
				{1., 3.},
				{2., 4.},
			},
			{
				{1., 3.},
				{2., 4.},
			},
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

		ten, err = qti.Zeros([]int32{5, 4, 3, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{5, 4, 2, 3}, conf)
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
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.Reshape([]int32{1, 1})
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.Zeros([]int32{1, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Reshape(nil)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 1, 1, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Reshape([]int32{1, 1})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{1, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Reshape([]int32{1, 4})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{1, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.Reshape([]int32{4, 1})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{4, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.Reshape([]int32{2, 2})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{2, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Reshape([]int32{6})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{6}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.Reshape([]int32{1, 6, 1})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{1, 6, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.Reshape([]int32{3, 2})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{3, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		shape := []int32{3, 2}

		act, err = ten.Reshape(shape)
		if err != nil {
			t.Fatal(err)
		}

		shape[0] = 1
		shape[1] = 6

		exp, err = qti.Zeros([]int32{3, 2}, conf)
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

func TestUnSqueeze(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.UnSqueeze(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.Zeros([]int32{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.UnSqueeze(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{1, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.UnSqueeze(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{2, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.UnSqueeze(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{1, 2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.UnSqueeze(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{2, 1, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.UnSqueeze(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{2, 3, 1}, conf)
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
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.Zeros([]int32{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.Squeeze(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Squeeze(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2, 1, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Squeeze(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2, 3, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Squeeze(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{2, 3}, conf)
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
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.Zeros([]int32{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.Flatten(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.Zeros([]int32{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Flatten(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{24}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Flatten(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{2, 12}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Flatten(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{2, 3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 2, 3, 4, 5, 6}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Flatten(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Zeros([]int32{1, 2, 360}, conf)
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
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.TensorOf(5., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.Broadcast(nil)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := qti.TensorOf(5., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf(5., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Broadcast([]int32{2, 1})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{{5.}, {5.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.TensorOf([]float64{5.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Broadcast([]int32{3, 2})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][]float64{
			{5., 5.},
			{5., 5.},
			{5., 5.},
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

		ten, err = qti.TensorOf([]float64{1., 2.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Broadcast([]int32{3, 3, 2})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][][]float64{
			{
				{1., 2.},
				{1., 2.},
				{1., 2.},
			},
			{
				{1., 2.},
				{1., 2.},
				{1., 2.},
			},
			{
				{1., 2.},
				{1., 2.},
				{1., 2.},
			},
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

		ten, err = qti.TensorOf([][][]float64{{{0.}}, {{1.}}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Broadcast([]int32{2, 3, 4})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][][]float64{
			{
				{0., 0., 0., 0.},
				{0., 0., 0., 0.},
				{0., 0., 0., 0.},
			},
			{
				{1., 1., 1., 1.},
				{1., 1., 1., 1.},
				{1., 1., 1., 1.},
			},
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

		ten, err = qti.TensorOf([][][]float64{
			{{0., 1., 2., 3.}},
			{{4., 5., 6., 7.}},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Broadcast([]int32{1, 2, 3, 4})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.TensorOf([][][][]float64{
			{
				{
					{0., 1., 2., 3.},
					{0., 1., 2., 3.},
					{0., 1., 2., 3.},
				},
				{
					{4., 5., 6., 7.},
					{4., 5., 6., 7.},
					{4., 5., 6., 7.},
				},
			},
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

		ten, err = qti.Ones([]int32{4, 1, 1, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Broadcast([]int32{6, 5, 4, 4, 3, 3, 3})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = qti.Ones([]int32{6, 5, 4, 4, 3, 3, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		shape := []int32{2, 3}

		act, err = ten.Broadcast([]int32{2, 3})
		if err != nil {
			t.Fatal(err)
		}

		shape[0] = 1
		shape[1] = 6

		exp, err = qti.Zeros([]int32{2, 3}, conf)
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

func TestValidationTranspose(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Transpose()
		if err == nil {
			t.Fatalf("expected error because of tensor having less than 2 dimensions")
		} else if err.Error() != "tensor's dimension validation failed: expected tensor to have at least (2) dimensions for transpose: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Transpose()
		if err == nil {
			t.Fatalf("expected error because of tensor having less than 2 dimensions")
		} else if err.Error() != "tensor's dimension validation failed: expected tensor to have at least (2) dimensions for transpose: got (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}

func TestValidationReshape(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.Zeros([]int32{3, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Reshape([]int32{2, 3, -1})
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		} else if err.Error() != "input shape validation failed: expected positive dimension sizes: got (-1) at position (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Reshape([]int32{2})
		if err == nil {
			t.Fatalf("expected error because of incompatible number of elements in source (1) with target (2)")
		} else if err.Error() != "input shape validation failed: expected number of elements in source and target tensors to match: (1) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Reshape([]int32{2, 3})
		if err == nil {
			t.Fatalf("expected error because of incompatible number of elements in source (2) with target (6)")
		} else if err.Error() != "input shape validation failed: expected number of elements in source and target tensors to match: (2) != (6)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}

func TestValidationUnSqueeze(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.UnSqueeze(-1)
		if err == nil {
			t.Fatalf("expected error because of dimension (-1) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,0]: got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.UnSqueeze(1)
		if err == nil {
			t.Fatalf("expected error because of dimension (1) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,0]: got (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.UnSqueeze(2)
		if err == nil {
			t.Fatalf("expected error because of dimension (2) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,1]: got (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}

func TestValidationSqueeze(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Squeeze(-1)
		if err == nil {
			t.Fatalf("expected error because of dimension (-1) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.Squeeze(0)
		if err == nil {
			t.Fatalf("expected error because of dimension (0) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Squeeze(1)
		if err == nil {
			t.Fatalf("expected error because of dimension (1) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Squeeze(3)
		if err == nil {
			t.Fatalf("expected error because of dimension (3) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,3): got (3)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.Squeeze(2)
		if err == nil {
			t.Fatalf("expected error because of dimension (2) not being equal to (1)")
		} else if err.Error() != "input dimension validation failed: expected squeeze dimension to be (1): got (3)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.Squeeze(1)
		if err == nil {
			t.Fatalf("expected error because of dimension (1) not being equal to (1)")
		} else if err.Error() != "input dimension validation failed: expected squeeze dimension to be (1): got (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}

func TestValidationFlatten(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Flatten(-1)
		if err == nil {
			t.Fatalf("expected error because of dimension (-1) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.Flatten(0)
		if err == nil {
			t.Fatalf("expected error because of dimension (0) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Flatten(1)
		if err == nil {
			t.Fatalf("expected error because of dimension (1) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Flatten(3)
		if err == nil {
			t.Fatalf("expected error because of dimension (3) being out of range")
		} else if err.Error() != "input dimension validation failed: expected dimension to be in range [0,3): got (3)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}

func TestValidationBroadcast(t *testing.T) {
	runTestLogicOnDevices(func(dev qti.Device) {

		conf := &qti.Config{Device: dev}

		/* ------------------------------ */

		ten, err := qti.Zeros([]int32{3, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Broadcast([]int32{3, -2})
		if err == nil {
			t.Fatalf("expected error because of negative dimension")
		} else if err.Error() != "input shape validation failed: expected positive dimension sizes: got (-2) at position (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{1, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Broadcast(nil)
		if err == nil {
			t.Fatalf("expected error because of source number of dimensions (2) being greater than that of target (0)")
		} else if err.Error() != "input shape validation failed: expected number of dimensions in source tensor to be less than or equal to that of target shape: (2) > (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.Broadcast([]int32{2})
		if err == nil {
			t.Fatalf("expected error because of source number of dimensions (2) being greater than that of target (1)")
		} else if err.Error() != "input shape validation failed: expected number of dimensions in source tensor to be less than or equal to that of target shape: (2) > (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Broadcast([]int32{1})
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (0)")
		} else if err.Error() != "input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.Broadcast([]int32{3})
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (0)")
		} else if err.Error() != "input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{4, 2, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Broadcast([]int32{4, 3, 5})
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		} else if err.Error() != "input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{2, 1, 4, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Broadcast([]int32{2, 3, 4, 4, 5})
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		} else if err.Error() != "input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = qti.Zeros([]int32{3, 1, 3, 6}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Broadcast([]int32{1, 2, 3, 4, 5, 6})
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (4)")
		} else if err.Error() != "input shape validation failed: expected target shape to be (3) or source size to be (1) at dimension (4): got shape (5)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
