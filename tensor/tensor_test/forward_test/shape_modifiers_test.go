package forward_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func TestTranspose(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, [][]float64{{1.}})
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, [][]float64{{1.}})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][]float64{{1., 0., 2.}})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{{1.}, {0.}, {2.}})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][]float64{{-2.}, {0.}, {-1.}})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{{-2., 0., -1.}})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][]float64{
			{0., 1., 2., 3.},
			{0., 1., 2., 3.},
			{0., 1., 2., 3.},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{0., 0., 0.},
			{1., 1., 1.},
			{2., 2., 2.},
			{3., 3., 3.},
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

		ten, err = tinit.TensorOf(conf, [][][]float64{
			{
				{1., 2.},
				{3., 4.},
			},
			{
				{1., 2.},
				{3., 4.},
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][]float64{
			{
				{1., 3.},
				{2., 4.},
			},
			{
				{1., 3.},
				{2., 4.},
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

		ten, err = tinit.Zeros(conf, 5, 4, 3, 2)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 5, 4, 2, 3)
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

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.Reshape(1, 1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.Zeros(conf, 1, 1)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Reshape()
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1, 1, 1, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Reshape(1, 1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 1, 1)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 4)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Reshape(1, 4)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 1, 4)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.Reshape(4, 1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 4, 1)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.Reshape(2, 2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 2, 2)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Reshape(6)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 6)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.Reshape(1, 6, 1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 1, 6, 1)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.Reshape(3, 2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 3, 2)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		shape := []int32{3, 2}

		act, err = ten.Reshape(shape...)
		if err != nil {
			t.Fatal(err)
		}

		shape[0] = 1
		shape[1] = 6

		exp, err = tinit.Zeros(conf, 3, 2)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.UnSqueeze(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.Zeros(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.UnSqueeze(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 1, 2)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.UnSqueeze(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 2, 1)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.UnSqueeze(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 1, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.UnSqueeze(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 2, 1, 3)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.UnSqueeze(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 2, 3, 1)
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

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.Zeros(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.Squeeze(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Squeeze(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2, 1, 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Squeeze(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2, 3, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Squeeze(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 2, 3)
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

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.Zeros(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.Flatten(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.Zeros(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Flatten(0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 24)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Flatten(1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 2, 12)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Flatten(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1, 2, 3, 4, 5, 6)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Flatten(2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 1, 2, 360)
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

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 5.)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.Broadcast()
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 5.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, 5.)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Broadcast(2, 1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{{5.}, {5.}})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{5.})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Broadcast(3, 2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{5., 5.},
			{5., 5.},
			{5., 5.},
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

		ten, err = tinit.TensorOf(conf, []float64{1., 2.})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Broadcast(3, 3, 2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][]float64{
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

		ten, err = tinit.TensorOf(conf, [][][]float64{{{0.}}, {{1.}}})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Broadcast(2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][]float64{
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

		ten, err = tinit.TensorOf(conf, [][][]float64{
			{{0., 1., 2., 3.}},
			{{4., 5., 6., 7.}},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Broadcast(1, 2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][][]float64{
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

		ten, err = tinit.Ones(conf, 4, 1, 1, 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Broadcast(6, 5, 4, 4, 3, 3, 3)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Ones(conf, 6, 5, 4, 4, 3, 3, 3)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2, 1)
		if err != nil {
			t.Fatal(err)
		}

		shape := []int32{2, 3}

		act, err = ten.Broadcast(2, 3)
		if err != nil {
			t.Fatal(err)
		}

		shape[0] = 1
		shape[1] = 6

		exp, err = tinit.Zeros(conf, 2, 3)
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
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Transpose()
		if err == nil {
			t.Fatalf("expected error because of tensor having less than 2 dimensions")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 3)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Transpose()
		if err == nil {
			t.Fatalf("expected error because of tensor having less than 2 dimensions")
		}

		/* ------------------------------ */

	})
}

func TestValidationReshape(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.Zeros(conf, 3, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Reshape(2, 3, -1)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Reshape(2)
		if err == nil {
			t.Fatalf("expected error because of incompatible number of elements in source (1) with target (2)")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Reshape(2, 3)
		if err == nil {
			t.Fatalf("expected error because of incompatible number of elements in source (1) with target (6)")
		}

		/* ------------------------------ */

	})
}

func TestValidationUnSqueeze(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.UnSqueeze(-1)
		if err == nil {
			t.Fatalf("expected error because of negative dimension")
		}

		_, err = ten.UnSqueeze(1)
		if err == nil {
			t.Fatalf("expected error because of dimension (1) being out of range")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.UnSqueeze(2)
		if err == nil {
			t.Fatalf("expected error because of dimension (2) being out of range")
		}

		/* ------------------------------ */

	})
}

func TestValidationSqueeze(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Squeeze(-1)
		if err == nil {
			t.Fatalf("expected error because of negative dimension")
		}

		_, err = ten.Squeeze(0)
		if err == nil {
			t.Fatalf("expected error because of dimension (0) being out of range")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Squeeze(1)
		if err == nil {
			t.Fatalf("expected error because of dimension (1) being out of range")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Squeeze(3)
		if err == nil {
			t.Fatalf("expected error because of dimension (3) being out of range")
		}

		_, err = ten.Squeeze(2)
		if err == nil {
			t.Fatalf("expected error because of dimension (2) not being equal to (1)")
		}

		_, err = ten.Squeeze(1)
		if err == nil {
			t.Fatalf("expected error because of dimension (1) not being equal to (1)")
		}

		/* ------------------------------ */

	})
}

func TestValidationFlatten(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Flatten(-1)
		if err == nil {
			t.Fatalf("expected error because of negative dimension")
		}

		_, err = ten.Flatten(0)
		if err == nil {
			t.Fatalf("expected error because of dimension (0) being out of range")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Flatten(1)
		if err == nil {
			t.Fatalf("expected error because of dimension (1) being out of range")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Flatten(3)
		if err == nil {
			t.Fatalf("expected error because of dimension (3) being out of range")
		}

		/* ------------------------------ */

	})
}

func TestValidationBroadcast(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.Zeros(conf, 3, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Broadcast(3, -2)
		if err == nil {
			t.Fatalf("expected error because of negative dimension")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Broadcast()
		if err == nil {
			t.Fatalf("expected error because of source number of dimensions (2) being greater than that of target (0)")
		}

		_, err = ten.Broadcast(2)
		if err == nil {
			t.Fatalf("expected error because of source number of dimensions (2) being greater than that of target (1)")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Broadcast(1)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (0)")
		}

		_, err = ten.Broadcast(3)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (0)")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 4, 2, 1)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Broadcast(4, 3, 5)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2, 1, 4, 1)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Broadcast(2, 3, 4, 4, 5)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (1)")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 3, 1, 3, 6)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Broadcast(1, 2, 3, 4, 5, 6)
		if err == nil {
			t.Fatalf("expected error because of incompatible sizes at dimension (2)")
		}

		/* ------------------------------ */

	})
}
