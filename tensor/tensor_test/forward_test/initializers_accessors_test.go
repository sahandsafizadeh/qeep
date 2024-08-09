package forward_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func TestFullEyeAt(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.Full(conf, -1.)
		if err != nil {
			t.Fatal(err)
		}

		if val, err := ten.At(); err != nil {
			t.Fatal(err)
		} else if int(val) != -1 {
			t.Fatalf("expected (-1) as scalar tensor value, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.Full(conf, 9., 1)
		if err != nil {
			t.Fatal(err)
		}

		if val, err := ten.At(0); err != nil {
			t.Fatal(err)
		} else if int(val) != 9 {
			t.Fatalf("expected (9) as tensor value in position [0], got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.Full(conf, 0., 1, 2)
		if err != nil {
			t.Fatal(err)
		}

		if val, err := ten.At(0, 0); err != nil {
			t.Fatal(err)
		} else if int(val) != 0 {
			t.Fatalf("expected (0) as tensor value in position [0,0], got (%f)", val)
		}

		if val, err := ten.At(0, 1); err != nil {
			t.Fatal(err)
		} else if int(val) != 0 {
			t.Fatalf("expected (0) as tensor value in position [0,1], got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.Full(conf, 5., 4, 3, 2, 1)
		if err != nil {
			t.Fatal(err)
		}

		var i, j, k, u int32
		for i = 0; i < 4; i++ {
			for j = 0; j < 3; j++ {
				for k = 0; k < 2; k++ {
					for u = 0; u < 1; u++ {
						if val, err := ten.At(i, j, k, u); err != nil {
							t.Fatal(err)
						} else if int(val) != 5 {
							t.Fatalf("expected (5) as tensor value in position [%d,%d,%d,%d], got (%f)", i, j, k, u, val)
						}
					}
				}
			}
		}

		/* ------------------------------ */

		ten, err = tinit.Eye(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		if val, err := ten.At(0, 0); err != nil {
			t.Fatal(err)
		} else if int(val) != 1 {
			t.Fatalf("expected (1) as eye tensor value in position [0,0], got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tinit.Eye(conf, 5)
		if err != nil {
			t.Fatal(err)
		}

		for i = 0; i < 5; i++ {
			for j = 0; j < 5; j++ {
				val, err := ten.At(i, j)
				if err != nil {
					t.Fatal(err)
				}

				if i == j {
					if int(val) != 1 {
						t.Fatalf("expected (1) as eye tensor value in position [%d,%d], got (%f)", i, j, val)
					}
				} else {
					if int(val) != 0 {
						t.Fatalf("expected (0) as eye tensor value in position [%d,%d], got (%f)", i, j, val)
					}
				}

			}
		}

		/* ------------------------------ */

		dims := []int32{3, 4}

		ten, err = tinit.Full(conf, 1., dims...)
		if err != nil {
			t.Fatal(err)
		}

		dims[0] = 1
		dims[1] = 1

		if val, err := ten.At(2, 3); err != nil {
			t.Fatal(err)
		} else if int(val) != 1 {
			t.Fatalf("expected (1) as tensor value in position [2,3], got (%f)", val)
		}

		/* ------------------------------ */

	})
}

func TestTensorOfSliceEquals(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.Slice(nil)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.TensorOf(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{3.})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Slice([]tensor.Range{{From: 0, To: 1}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{3.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{4.})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Slice([]tensor.Range{{}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{4.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{1., 4.})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Slice([]tensor.Range{{From: 0, To: 1}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{1.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.Slice([]tensor.Range{{From: 1, To: 2}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{4.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.Slice([]tensor.Range{{From: 0, To: 2}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{1., 4.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, [][]float64{{-1.}, {-2.}})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Slice([]tensor.Range{{From: 0, To: 1}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{{-1.}})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.Slice([]tensor.Range{{From: 1, To: 2}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{{-2.}})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		act, err = ten.Slice([]tensor.Range{{}, {From: 0, To: 1}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{{-1.}, {-2.}})
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
				{-1., 9., -5.},
				{2., 4., 6.},
				{0., 1., 2.},
			},
			{
				{1., -3., -7.},
				{9., 7., 5.},
				{6., 3., 9.},
			},
			{
				{-2., 9., -2.},
				{1., 2., 6.},
				{0., 1., 0.},
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Slice([]tensor.Range{{}, {From: 1, To: 2}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][]float64{
			{{2., 4., 6.}},
			{{9., 7., 5.}},
			{{1., 2., 6.}},
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

		act, err = ten.Slice([]tensor.Range{{From: 0, To: 2}, {}, {From: 1, To: 3}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][]float64{
			{
				{9., -5.},
				{4., 6.},
				{1., 2.},
			},
			{
				{-3., -7.},
				{7., 5.},
				{3., 9.},
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

		ten, err = tinit.TensorOf(conf, [][][][]float64{
			{
				{
					{1., 2., 3., 4.},
					{1., 2., 3., 4.},
					{1., 2., 3., 4.},
				},
				{
					{1., 2., 3., 4.},
					{1., 2., 3., 4.},
					{1., 2., 3., 4.},
				},
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Slice([]tensor.Range{{}, {From: 1, To: 2}, {From: 1, To: 3}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][][]float64{
			{
				{
					{1., 2., 3., 4.},
					{1., 2., 3., 4.},
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

		d1 := []float64{5.}

		ten, err = tinit.TensorOf(conf, d1)
		if err != nil {
			t.Fatal(err)
		}

		d1[0] = 3.

		act, err = ten.Slice(nil)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{5.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		d4 := [][][][]float64{{{{5.}}}}

		ten, err = tinit.TensorOf(conf, d4)
		if err != nil {
			t.Fatal(err)
		}

		d4[0][0][0][0] = 3.

		act, err = ten.Slice(nil)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][][]float64{{{{5.}}}})
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

func TestZerosOnesPatch(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.Ones(conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Patch(nil, t2)
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

		t1, err = tinit.Zeros(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Patch([]tensor.Range{{From: 1, To: 2}}, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, []float64{0., 1.})
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 3, 3)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 2, 2)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act, err = t1.Patch([]tensor.Range{{From: 0, To: 2}, {From: 0, To: 2}}, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{1., 1., 0.},
			{1., 1., 0.},
			{0., 0., 0.},
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

		act, err = t1.Patch([]tensor.Range{{From: 0, To: 2}, {From: 1, To: 3}}, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{0., 1., 1.},
			{0., 1., 1.},
			{0., 0., 0.},
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

		act, err = t1.Patch([]tensor.Range{{From: 1, To: 3}, {From: 0, To: 2}}, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{0., 0., 0.},
			{1., 1., 0.},
			{1., 1., 0.},
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

		act, err = t1.Patch([]tensor.Range{{From: 1, To: 3}, {From: 1, To: 3}}, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][]float64{
			{0., 0., 0.},
			{0., 1., 1.},
			{0., 1., 1.},
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

		t1, err = tinit.Zeros(conf, 4, 3, 2)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 3, 2, 1)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Patch(nil, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][]float64{
			{
				{1., 0.},
				{1., 0.},
				{0., 0.},
			},
			{
				{1., 0.},
				{1., 0.},
				{0., 0.},
			},
			{
				{1., 0.},
				{1., 0.},
				{0., 0.},
			},
			{
				{0., 0.},
				{0., 0.},
				{0., 0.},
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

func TestRandoms(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		dims := []int32{3, 4}

		ten, err := tinit.RandU(conf, -1., 1., dims...)
		if err != nil {
			t.Fatal(err)
		}

		dims[0] = 1
		dims[1] = 1

		shape := ten.Shape()
		if !shapesEqual(shape, []int32{3, 4}) {
			t.Fatalf("expected tensor to have shape [3, 4], got %v", shape)
		}

		/* ------------------------------ */

		dims = []int32{3, 4}

		ten, err = tinit.RandN(conf, 0., 1., dims...)
		if err != nil {
			t.Fatal(err)
		}

		dims[0] = 1
		dims[1] = 1

		shape = ten.Shape()
		if !shapesEqual(shape, []int32{3, 4}) {
			t.Fatalf("expected tensor to have shape [3, 4], got %v", shape)
		}

		/* ------------------------------ */

	})
}

func TestConcat(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.Zeros(conf, 3)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.Zeros(conf, 5)
		if err != nil {
			t.Fatal(err)
		}

		act, err := tinit.Concat([]tensor.Tensor{t1, t2}, 0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tinit.Zeros(conf, 8)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 1, 5, 3)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 3, 5, 3)
		if err != nil {
			t.Fatal(err)
		}

		t3, err := tinit.Zeros(conf, 2, 5, 3)
		if err != nil {
			t.Fatal(err)
		}

		t4, err := tinit.Zeros(conf, 4, 5, 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = tinit.Concat([]tensor.Tensor{t1, t2, t3, t4}, 0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 10, 5, 3)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 4, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 4, 4, 3)
		if err != nil {
			t.Fatal(err)
		}

		t3, err = tinit.Zeros(conf, 4, 1, 3)
		if err != nil {
			t.Fatal(err)
		}

		t4, err = tinit.Zeros(conf, 4, 3, 3)
		if err != nil {
			t.Fatal(err)
		}

		act, err = tinit.Concat([]tensor.Tensor{t1, t2, t3, t4}, 1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 4, 10, 3)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tinit.TensorOf(conf, [][][]float64{
			{
				{0., 1., 2.},
				{3., 4., 5.},
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		t2, err = t1.Slice(nil)
		if err != nil {
			t.Fatal(err)
		}

		t3, err = t1.Slice(nil)
		if err != nil {
			t.Fatal(err)
		}

		act, err = tinit.Concat([]tensor.Tensor{t1, t2, t3}, 0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][]float64{
			{
				{0., 1., 2.},
				{3., 4., 5.},
			},
			{
				{0., 1., 2.},
				{3., 4., 5.},
			},
			{
				{0., 1., 2.},
				{3., 4., 5.},
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

		/* --------------- */

		act, err = tinit.Concat([]tensor.Tensor{t1, t2, t3}, 1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][]float64{
			{
				{0., 1., 2.},
				{3., 4., 5.},
				{0., 1., 2.},
				{3., 4., 5.},
				{0., 1., 2.},
				{3., 4., 5.},
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

		/* --------------- */

		act, err = tinit.Concat([]tensor.Tensor{t1, t2, t3}, 2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.TensorOf(conf, [][][]float64{
			{
				{0., 1., 2., 0., 1., 2., 0., 1., 2.},
				{3., 4., 5., 3., 4., 5., 3., 4., 5.},
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

		t1, err = tinit.Zeros(conf, 4)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 6)
		if err != nil {
			t.Fatal(err)
		}

		ts := []tensor.Tensor{t1, t2}

		act, err = tinit.Concat(ts, 0)
		if err != nil {
			t.Fatal(err)
		}

		ts[1], err = tinit.Ones(conf, 6)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tinit.Zeros(conf, 10)
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

func TestNElemsShape(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		nElems := ten.NElems()
		if nElems != 1 {
			t.Fatalf("expected tensor to have (1) element, got (%d)", nElems)
		}

		shape := ten.Shape()
		if !shapesEqual(shape, []int32{}) {
			t.Fatalf("expected tensor to have shape [], got %v", shape)
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		nElems = ten.NElems()
		if nElems != 1 {
			t.Fatalf("expected tensor to have (1) element, got (%d)", nElems)
		}

		shape = ten.Shape()
		if !shapesEqual(shape, []int32{1}) {
			t.Fatalf("expected tensor to have shape [1], got %v", shape)
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		nElems = ten.NElems()
		if nElems != 2 {
			t.Fatalf("expected tensor to have (2) element, got (%d)", nElems)
		}

		shape = ten.Shape()
		if !shapesEqual(shape, []int32{2}) {
			t.Fatalf("expected tensor to have shape [2], got %v", shape)
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 3, 4)
		if err != nil {
			t.Fatal(err)
		}

		nElems = ten.NElems()
		if nElems != 12 {
			t.Fatalf("expected tensor to have (12) element, got (%d)", nElems)
		}

		shape = ten.Shape()
		if !shapesEqual(shape, []int32{3, 4}) {
			t.Fatalf("expected tensor to have shape [3, 4], got %v", shape)
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 5, 4, 3, 2, 1)
		if err != nil {
			t.Fatal(err)
		}

		nElems = ten.NElems()
		if nElems != 120 {
			t.Fatalf("expected tensor to have (120) element, got (%d)", nElems)
		}

		shape = ten.Shape()
		if !shapesEqual(shape, []int32{5, 4, 3, 2, 1}) {
			t.Fatalf("expected tensor to have shape [5, 4, 3, 2, 1], got %v", shape)
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		shape = ten.Shape()
		shape[0] = 1
		shape[1] = 1

		shape = ten.Shape()
		if !shapesEqual(shape, []int32{2, 3}) {
			t.Fatalf("expected tensor to have shape [2, 3], got %v", shape)
		}

		/* ------------------------------ */

	})
}

func TestValidationFull(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		_, err := tinit.Full(conf, 2., -1)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		}

		_, err = tinit.Full(conf, 2., 0)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		}

		_, err = tinit.Full(conf, 2., 1, -2)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		}

		_, err = tinit.Full(conf, 2., 2, 0, 1)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		}

		/* ------------------------------ */

	})
}

func TestValidationRandU(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		_, err := tinit.RandU(conf, 0., -1.)
		if err == nil {
			t.Fatalf("expected error because of lower bound not being less than upper bound")
		}

		_, err = tinit.RandU(conf, 1., 1.)
		if err == nil {
			t.Fatalf("expected error because of lower bound not being less than upper bound")
		}

		_, err = tinit.RandU(conf, -1., 1., -1)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		}

		/* ------------------------------ */

	})
}

func TestValidationRandN(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		_, err := tinit.RandN(conf, 0., -1.)
		if err == nil {
			t.Fatalf("expected error because of non-positive standard deviation")
		}

		_, err = tinit.RandN(conf, -1., 0.)
		if err == nil {
			t.Fatalf("expected error because of non-positive standard deviation")
		}

		_, err = tinit.RandN(conf, 0., 1., -1)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		}

		/* ------------------------------ */

	})
}

func TestValidationTensorOf(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		_, err := tinit.TensorOf(conf, []float64{})
		if err == nil {
			t.Fatalf("expected error because of zero len along dimension (0)")
		}

		_, err = tinit.TensorOf(conf, [][]float64{{}, {}})
		if err == nil {
			t.Fatalf("expected error because of zero len along dimension (1)")
		}

		_, err = tinit.TensorOf(conf, [][]float64{{-1.}, {}})
		if err == nil {
			t.Fatalf("expected error because of zero len along dimension (1)")
		}

		_, err = tinit.TensorOf(conf, [][]float64{{}, {-2.}})
		if err == nil {
			t.Fatalf("expected error because of zero len along dimension (1)")
		}

		/* ------------------------------ */

		_, err = tinit.TensorOf(conf, [][][]float64{
			{
				{2., 2., 2.},
				{2., 2., 2.},
				{2., 2., 2.},
			},
			{
				{2., 2., 2.},
				{2., 2., 2.},
				{2., 2., 2.},
			},
			{
				{2., 2., 2.},
				{2., 2., 2.},
				{2., 2.},
			},
		})
		if err == nil {
			t.Fatalf("expected error because of inconsistent tensor len along dimension (2)")
		}

		/* ------------------------------ */

		_, err = tinit.TensorOf(conf, [][][]float64{
			{
				{3., 3., 3.},
				{3., 3., 3.},
				{3., 3., 3.},
			},
			{
				{3., 3., 3.},
				{3., 3., 3.},
			},
			{
				{3., 3., 3.},
				{3., 3., 3.},
				{3., 3., 3.},
			},
		})
		if err == nil {
			t.Fatalf("expected error because of inconsistent tensor len along dimension (1)")
		}

		/* ------------------------------ */

	})
}

func TestValidationConcat(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		_, err := tinit.Concat(nil, 0)
		if err == nil {
			t.Fatalf("expected error because of the number of input tensors being less than (2)")
		}

		_, err = tinit.Concat([]tensor.Tensor{nil}, 0)
		if err == nil {
			t.Fatalf("expected error because of the number of input tensors being less than (2)")
		}

		/* ------------------------------ */

		t1, err := tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.Zeros(conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = tinit.Concat([]tensor.Tensor{t1, t2}, 0)
		if err == nil {
			t.Fatalf("expected error because of having scalar tensors as input")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		t3, err := tinit.Zeros(conf, 2, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = tinit.Concat([]tensor.Tensor{t1, t2, t3}, 0)
		if err == nil {
			t.Fatalf("expected error because of the input tensors not having equal number of dimensions")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 3)
		if err != nil {
			t.Fatal(err)
		}

		_, err = tinit.Concat([]tensor.Tensor{t1, t2}, -1)
		if err == nil {
			t.Fatalf("expected error because of negative dimension")
		}

		_, err = tinit.Concat([]tensor.Tensor{t1, t2}, 1)
		if err == nil {
			t.Fatalf("expected error because of dimension (1) being out of range")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 3, 3)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 3, 3)
		if err != nil {
			t.Fatal(err)
		}

		_, err = tinit.Concat([]tensor.Tensor{t1, t2}, 2)
		if err == nil {
			t.Fatalf("expected error because of dimension (2) being out of range")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 2, 2, 2)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 2, 2, 1)
		if err != nil {
			t.Fatal(err)
		}

		t3, err = tinit.Zeros(conf, 3, 2, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = tinit.Concat([]tensor.Tensor{t1, t2, t3}, 0)
		if err == nil {
			t.Fatalf("expected error because of size mismatch along dimension (2)")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 2, 1, 2)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 2, 2, 2)
		if err != nil {
			t.Fatal(err)
		}

		t3, err = tinit.Zeros(conf, 3, 2, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = tinit.Concat([]tensor.Tensor{t1, t2, t3}, 0)
		if err == nil {
			t.Fatalf("expected error because of size mismatch along dimension (1)")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 2, 1, 2)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Zeros(conf, 1, 2, 2)
		if err != nil {
			t.Fatal(err)
		}

		t3, err = tinit.Zeros(conf, 2, 3, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = tinit.Concat([]tensor.Tensor{t1, t2, t3}, 1)
		if err == nil {
			t.Fatalf("expected error because of size mismatch along dimension (0)")
		}

		/* ------------------------------ */

	})
}

func TestValidationAt(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.Zeros(conf, 1)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.At()
		if err == nil {
			t.Fatalf("expected error because of incompatible index len (0) with dimension len (1)")
		}

		_, err = ten.At(0, 0)
		if err == nil {
			t.Fatalf("expected error because of incompatible index len (2) with dimension len (1)")
		}

		_, err = ten.At(-1)
		if err == nil {
			t.Fatalf("expected error because of negative index")
		}

		_, err = ten.At(1)
		if err == nil {
			t.Fatalf("expected error because of index (1) at dimension (0) being out of range [0,1)")
		}

		/* ------------------------------ */

		ten, err = tinit.Zeros(conf, 1, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.At(0)
		if err == nil {
			t.Fatalf("expected error because of incompatible index len (1) with dimension len (2)")
		}

		_, err = ten.At(0, 1, 0)
		if err == nil {
			t.Fatalf("expected error because of incompatible index len (3) with dimension len (2)")
		}

		_, err = ten.At(-2, -1)
		if err == nil {
			t.Fatalf("expected error because of negative index")
		}

		_, err = ten.At(1, 0)
		if err == nil {
			t.Fatalf("expected error because of index (1) at dimension (0) being out of range [0,1)")
		}

		_, err = ten.At(1, 2)
		if err == nil {
			t.Fatalf("expected error because of index (2) at dimension (1) being out of range [0,2)")
		}

		/* ------------------------------ */

	})
}

func TestValidationSlice(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tinit.TensorOf(conf, 2.)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Slice([]tensor.Range{{From: 0, To: 0}})
		if err == nil {
			t.Fatalf("expected error because of incompatible index len (1) with dimension len (0)")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{3.})
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Slice([]tensor.Range{{From: 0, To: 0}, {From: 0, To: 0}})
		if err == nil {
			t.Fatalf("expected error because of incompatible index len (2) with dimension len (1)")
		}

		_, err = ten.Slice([]tensor.Range{{From: 1, To: 1}})
		if err == nil {
			t.Fatalf("expected error because of to index (0) not being larger than from index (0)")
		}

		_, err = ten.Slice([]tensor.Range{{From: -1, To: 0}})
		if err == nil {
			t.Fatalf("expected error because of negative from index (-1)")
		}

		_, err = ten.Slice([]tensor.Range{{From: 1, To: 2}})
		if err == nil {
			t.Fatalf("expected error because of from index (1) being out of range [0,1) at dimension (0)")
		}

		_, err = ten.Slice([]tensor.Range{{From: 0, To: 2}})
		if err == nil {
			t.Fatalf("expected error because of to index (2) being out of range [0,1) at dimension (0)")
		}

		/* ------------------------------ */

		ten, err = tinit.TensorOf(conf, []float64{1., 4.})
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Slice([]tensor.Range{{From: 2, To: 3}})
		if err == nil {
			t.Fatalf("expected error because of from index (2) being out of range [0,2) at dimension (0)")
		}

		_, err = ten.Slice([]tensor.Range{{From: 1, To: 3}})
		if err == nil {
			t.Fatalf("expected error because of to index (3) being out of range [0,2) at dimension (0)")
		}

		/* ------------------------------ */

	})
}

func TestValidationPatch(t *testing.T) {
	runTestLogicOnDevices(func(dev tinit.Device) {

		conf := &tinit.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tinit.Zeros(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tinit.Ones(conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Patch([]tensor.Range{}, t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible number of dimensions")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 1, 1)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 1, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Patch([]tensor.Range{}, t2)
		if err == nil {
			t.Fatalf("expected error because of exceeding patch size at dimension (1)")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 3)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Patch([]tensor.Range{{From: 2, To: 4}}, t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible index with target tensor")
		}

		/* ------------------------------ */

		t1, err = tinit.Zeros(conf, 1, 3)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tinit.Ones(conf, 1, 2)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Patch([]tensor.Range{{}, {From: 2, To: 3}}, t2)
		if err == nil {
			t.Fatalf("expected error because of index not covering source tensor at dimension (1)")
		}

		/* ------------------------------ */

	})
}

/* ----- helpers ----- */

func shapesEqual(s1, s2 []int32) (ok bool) {
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
