package forward_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestFullEyeAt(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tensor.Full(nil, -1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if val, err := ten.At(); err != nil {
			t.Fatal(err)
		} else if int(val) != -1 {
			t.Fatalf("expected (-1) as scalar tensor value, got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.Full([]int{1}, 9., conf)
		if err != nil {
			t.Fatal(err)
		}

		if val, err := ten.At(0); err != nil {
			t.Fatal(err)
		} else if int(val) != 9 {
			t.Fatalf("expected (9) as tensor value in position [0], got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.Full([]int{1, 2}, 0., conf)
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

		ten, err = tensor.Full([]int{4, 3, 2, 1}, 5., conf)
		if err != nil {
			t.Fatal(err)
		}

		var i, j, k, u int
		for i = range 4 {
			for j = range 3 {
				for k = range 2 {
					for u = range 1 {
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

		ten, err = tensor.Eye(1, conf)
		if err != nil {
			t.Fatal(err)
		}

		if val, err := ten.At(0, 0); err != nil {
			t.Fatal(err)
		} else if int(val) != 1 {
			t.Fatalf("expected (1) as eye tensor value in position [0,0], got (%f)", val)
		}

		/* ------------------------------ */

		ten, err = tensor.Eye(5, conf)
		if err != nil {
			t.Fatal(err)
		}

		for i = range 5 {
			for j = range 5 {
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

		dims := []int{3, 4}

		ten, err = tensor.Full(dims, 1., conf)
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

func TestOfSliceEquals(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tensor.Of(2., conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := ten.Slice(nil)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.Of(2., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Of([]float64{3.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Slice([]tensor.Range{{From: 0, To: 1}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([]float64{3.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Of([]float64{4.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Slice([]tensor.Range{{}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([]float64{4.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Of([]float64{1., 4.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Slice([]tensor.Range{{From: 0, To: 1}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([]float64{1.}, conf)
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

		exp, err = tensor.Of([]float64{4.}, conf)
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

		exp, err = tensor.Of([]float64{1., 4.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Of([][]float64{{-1.}, {-2.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Slice([]tensor.Range{{From: 0, To: 1}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][]float64{{-1.}}, conf)
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

		exp, err = tensor.Of([][]float64{{-2.}}, conf)
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

		exp, err = tensor.Of([][]float64{{-1.}, {-2.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ten, err = tensor.Of([][][]float64{
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
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Slice([]tensor.Range{{}, {From: 1, To: 2}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][][]float64{
			{{2., 4., 6.}},
			{{9., 7., 5.}},
			{{1., 2., 6.}},
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

		act, err = ten.Slice([]tensor.Range{{From: 0, To: 2}, {}, {From: 1, To: 3}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][][]float64{
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

		ten, err = tensor.Of([][][][]float64{
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
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = ten.Slice([]tensor.Range{{}, {From: 1, To: 2}, {From: 1, To: 3}})
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][][][]float64{
			{
				{
					{1., 2., 3., 4.},
					{1., 2., 3., 4.},
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

		d1 := []float64{5.}

		ten, err = tensor.Of(d1, conf)
		if err != nil {
			t.Fatal(err)
		}

		d1[0] = 3.

		act, err = ten.Slice(nil)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([]float64{5.}, conf)
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

		ten, err = tensor.Of(d4, conf)
		if err != nil {
			t.Fatal(err)
		}

		d4[0][0][0][0] = 3.

		act, err = ten.Slice(nil)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][][][]float64{{{{5.}}}}, conf)
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
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tensor.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tensor.Ones(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := t1.Patch(nil, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.Of(1., conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tensor.Zeros([]int{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Ones([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = t1.Patch([]tensor.Range{{From: 1, To: 2}}, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([]float64{0., 1.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tensor.Zeros([]int{3, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Ones([]int{2, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act, err = t1.Patch([]tensor.Range{{From: 0, To: 2}, {From: 0, To: 2}}, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][]float64{
			{1., 1., 0.},
			{1., 1., 0.},
			{0., 0., 0.},
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

		act, err = t1.Patch([]tensor.Range{{From: 0, To: 2}, {From: 1, To: 3}}, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][]float64{
			{0., 1., 1.},
			{0., 1., 1.},
			{0., 0., 0.},
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

		act, err = t1.Patch([]tensor.Range{{From: 1, To: 3}, {From: 0, To: 2}}, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][]float64{
			{0., 0., 0.},
			{1., 1., 0.},
			{1., 1., 0.},
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

		act, err = t1.Patch([]tensor.Range{{From: 1, To: 3}, {From: 1, To: 3}}, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][]float64{
			{0., 0., 0.},
			{0., 1., 1.},
			{0., 1., 1.},
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

		t1, err = tensor.Zeros([]int{4, 3, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Ones([]int{3, 2, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act, err = t1.Patch(nil, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][][]float64{
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

		act, err = t1.Patch([]tensor.Range{{From: 1, To: 4}}, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][][]float64{
			{
				{0., 0.},
				{0., 0.},
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
				{1., 0.},
				{1., 0.},
				{0., 0.},
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

		/* --------------- */

		act, err = t1.Patch([]tensor.Range{{From: 1, To: 4}, {From: 1, To: 3}, {From: 1, To: 2}}, t2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][][]float64{
			{
				{0., 0.},
				{0., 0.},
				{0., 0.},
			},
			{
				{0., 0.},
				{0., 1.},
				{0., 1.},
			},
			{
				{0., 0.},
				{0., 1.},
				{0., 1.},
			},
			{
				{0., 0.},
				{0., 1.},
				{0., 1.},
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

	})
}

func TestRandoms(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		dims := []int{3, 4}

		ten, err := tensor.RandU(dims, -1., 1., conf)
		if err != nil {
			t.Fatal(err)
		}

		dims[0] = 1
		dims[1] = 1

		shape := ten.Shape()
		if !slices.Equal(shape, []int{3, 4}) {
			t.Fatalf("expected tensor to have shape [3, 4], got %v", shape)
		}

		/* ------------------------------ */

		dims = []int{3, 4}

		ten, err = tensor.RandN(dims, 0., 1., conf)
		if err != nil {
			t.Fatal(err)
		}

		dims[0] = 1
		dims[1] = 1

		shape = ten.Shape()
		if !slices.Equal(shape, []int{3, 4}) {
			t.Fatalf("expected tensor to have shape [3, 4], got %v", shape)
		}

		/* ------------------------------ */

	})
}

func TestConcat(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tensor.Zeros([]int{3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tensor.Zeros([]int{5}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := tensor.Concat([]tensor.Tensor{t1, t2}, 0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.Zeros([]int{8}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tensor.Zeros([]int{1, 5, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Zeros([]int{3, 5, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t3, err := tensor.Zeros([]int{2, 5, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t4, err := tensor.Zeros([]int{4, 5, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = tensor.Concat([]tensor.Tensor{t1, t2, t3, t4}, 0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Zeros([]int{10, 5, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tensor.Zeros([]int{4, 2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Zeros([]int{4, 4, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t3, err = tensor.Zeros([]int{4, 1, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t4, err = tensor.Zeros([]int{4, 3, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = tensor.Concat([]tensor.Tensor{t1, t2, t3, t4}, 1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Zeros([]int{4, 10, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		t1, err = tensor.Of([][][]float64{
			{
				{0., 1., 2.},
				{3., 4., 5.},
			},
		}, conf)
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

		act, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 0)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][][]float64{
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

		act, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 1)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][][]float64{
			{
				{0., 1., 2.},
				{3., 4., 5.},
				{0., 1., 2.},
				{3., 4., 5.},
				{0., 1., 2.},
				{3., 4., 5.},
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

		/* --------------- */

		act, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 2)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][][]float64{
			{
				{0., 1., 2., 0., 1., 2., 0., 1., 2.},
				{3., 4., 5., 3., 4., 5., 3., 4., 5.},
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

		t1, err = tensor.Zeros([]int{4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Zeros([]int{6}, conf)
		if err != nil {
			t.Fatal(err)
		}

		ts := []tensor.Tensor{t1, t2}

		act, err = tensor.Concat(ts, 0)
		if err != nil {
			t.Fatal(err)
		}

		ts[1], err = tensor.Ones([]int{6}, conf)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Zeros([]int{10}, conf)
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
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tensor.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		nElems := ten.NElems()
		if nElems != 1 {
			t.Fatalf("expected tensor to have (1) element, got (%d)", nElems)
		}

		shape := ten.Shape()
		if !slices.Equal(shape, []int{}) {
			t.Fatalf("expected tensor to have shape [], got %v", shape)
		}

		/* ------------------------------ */

		ten, err = tensor.Zeros([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		nElems = ten.NElems()
		if nElems != 1 {
			t.Fatalf("expected tensor to have (1) element, got (%d)", nElems)
		}

		shape = ten.Shape()
		if !slices.Equal(shape, []int{1}) {
			t.Fatalf("expected tensor to have shape [1], got %v", shape)
		}

		/* ------------------------------ */

		ten, err = tensor.Zeros([]int{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		nElems = ten.NElems()
		if nElems != 2 {
			t.Fatalf("expected tensor to have (2) element, got (%d)", nElems)
		}

		shape = ten.Shape()
		if !slices.Equal(shape, []int{2}) {
			t.Fatalf("expected tensor to have shape [2], got %v", shape)
		}

		/* ------------------------------ */

		ten, err = tensor.Zeros([]int{3, 4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		nElems = ten.NElems()
		if nElems != 12 {
			t.Fatalf("expected tensor to have (12) element, got (%d)", nElems)
		}

		shape = ten.Shape()
		if !slices.Equal(shape, []int{3, 4}) {
			t.Fatalf("expected tensor to have shape [3, 4], got %v", shape)
		}

		/* ------------------------------ */

		ten, err = tensor.Zeros([]int{5, 4, 3, 2, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		nElems = ten.NElems()
		if nElems != 120 {
			t.Fatalf("expected tensor to have (120) element, got (%d)", nElems)
		}

		shape = ten.Shape()
		if !slices.Equal(shape, []int{5, 4, 3, 2, 1}) {
			t.Fatalf("expected tensor to have shape [5, 4, 3, 2, 1], got %v", shape)
		}

		/* ------------------------------ */

		ten, err = tensor.Zeros([]int{2, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		shape = ten.Shape()
		shape[0] = 1
		shape[1] = 1

		shape = ten.Shape()
		if !slices.Equal(shape, []int{2, 3}) {
			t.Fatalf("expected tensor to have shape [2, 3], got %v", shape)
		}

		/* ------------------------------ */

	})
}

func TestValidationFullZerosOnes(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		_, err := tensor.Full([]int{-1}, 2., conf)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		} else if err.Error() != "Full input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Full([]int{0}, 2., conf)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		} else if err.Error() != "Full input dimension validation failed: expected positive dimension sizes: got (0) at position (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Full([]int{1, -2}, 2., conf)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		} else if err.Error() != "Full input dimension validation failed: expected positive dimension sizes: got (-2) at position (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Full([]int{2, 0, 1}, 2., conf)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		} else if err.Error() != "Full input dimension validation failed: expected positive dimension sizes: got (0) at position (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Zeros([]int{2, 0, 1}, conf)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		} else if err.Error() != "Zeros input dimension validation failed: expected positive dimension sizes: got (0) at position (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Ones([]int{2, 0, 1}, conf)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		} else if err.Error() != "Ones input dimension validation failed: expected positive dimension sizes: got (0) at position (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		conf = &tensor.Config{Device: -1}

		/* ------------------------------ */

		_, err = tensor.Full(nil, 2., conf)
		if err == nil {
			t.Fatalf("expected error because of invalid input device")
		} else if err.Error() != "Full tensor config data validation failed: invalid input device" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Zeros(nil, conf)
		if err == nil {
			t.Fatalf("expected error because of invalid input device")
		} else if err.Error() != "Zeros tensor config data validation failed: invalid input device" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Ones(nil, conf)
		if err == nil {
			t.Fatalf("expected error because of invalid input device")
		} else if err.Error() != "Ones tensor config data validation failed: invalid input device" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}

func TestValidationEye(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		_, err := tensor.Eye(-1, conf)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		} else if err.Error() != "Eye input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Eye(0, conf)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		} else if err.Error() != "Eye input dimension validation failed: expected positive dimension sizes: got (0) at position (0)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		conf = &tensor.Config{Device: -1}

		/* ------------------------------ */

		_, err = tensor.Eye(1, conf)
		if err == nil {
			t.Fatalf("expected error because of invalid input device")
		} else if err.Error() != "Eye tensor config data validation failed: invalid input device" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}

func TestValidationRandU(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		_, err := tensor.RandU(nil, 0., -1., conf)
		if err == nil {
			t.Fatalf("expected error because of lower bound not being less than upper bound")
		} else if err.Error() != "RandU random parameter validation failed: expected uniform random lower bound to be less than the upper bound: (0.000000) >= (-1.000000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.RandU(nil, 1., 1., conf)
		if err == nil {
			t.Fatalf("expected error because of lower bound not being less than upper bound")
		} else if err.Error() != "RandU random parameter validation failed: expected uniform random lower bound to be less than the upper bound: (1.000000) >= (1.000000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.RandU([]int{-1}, -1., 1., conf)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		} else if err.Error() != "RandU input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		conf = &tensor.Config{Device: -1}

		/* ------------------------------ */

		_, err = tensor.RandU(nil, 0., 1., conf)
		if err == nil {
			t.Fatalf("expected error because of invalid input device")
		} else if err.Error() != "RandU tensor config data validation failed: invalid input device" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}

func TestValidationRandN(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		_, err := tensor.RandN(nil, 0., -1., conf)
		if err == nil {
			t.Fatalf("expected error because of non-positive standard deviation")
		} else if err.Error() != "RandN random parameter validation failed: expected normal random standard deviation to be positive: got (-1.000000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.RandN(nil, -1., 0., conf)
		if err == nil {
			t.Fatalf("expected error because of non-positive standard deviation")
		} else if err.Error() != "RandN random parameter validation failed: expected normal random standard deviation to be positive: got (0.000000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.RandN([]int{-1}, 0., 1., conf)
		if err == nil {
			t.Fatalf("expected error because of non-positive dimension")
		} else if err.Error() != "RandN input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		conf = &tensor.Config{Device: -1}

		/* ------------------------------ */

		_, err = tensor.RandN(nil, 0., 1., conf)
		if err == nil {
			t.Fatalf("expected error because of invalid input device")
		} else if err.Error() != "RandN tensor config data validation failed: invalid input device" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}

func TestValidationOf(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		_, err := tensor.Of([]float64{}, conf)
		if err == nil {
			t.Fatalf("expected error because of zero len along dimension (0)")
		} else if err.Error() != "Of input data validation failed: expected data to not have zero length along any dimension" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Of([][]float64{}, conf)
		if err == nil {
			t.Fatalf("expected error because of zero len along dimension (0)")
		} else if err.Error() != "Of input data validation failed: expected data to not have zero length along any dimension" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Of([][][]float64{}, conf)
		if err == nil {
			t.Fatalf("expected error because of zero len along dimension (0)")
		} else if err.Error() != "Of input data validation failed: expected data to not have zero length along any dimension" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Of([][][][]float64{}, conf)
		if err == nil {
			t.Fatalf("expected error because of zero len along dimension (0)")
		} else if err.Error() != "Of input data validation failed: expected data to not have zero length along any dimension" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		_, err = tensor.Of([][]float64{{}, {}}, conf)
		if err == nil {
			t.Fatalf("expected error because of zero len along dimension (1)")
		} else if err.Error() != "Of input data validation failed: expected data to not have zero length along any dimension" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Of([][]float64{{}, {-1.}}, conf)
		if err == nil {
			t.Fatalf("expected error because of zero len along dimension (1)")
		} else if err.Error() != "Of input data validation failed: expected data to not have zero length along any dimension" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Of([][][]float64{{{}}}, conf)
		if err == nil {
			t.Fatalf("expected error because of zero len along dimension (1)")
		} else if err.Error() != "Of input data validation failed: expected data to not have zero length along any dimension" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		_, err = tensor.Of([][][]float64{
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
		}, conf)
		if err == nil {
			t.Fatalf("expected error because of inconsistent tensor len along dimension (2)")
		} else if err.Error() != "Of input data validation failed: expected data to have have equal length along every dimension" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Of([][][][]float64{
			{
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
			},
		}, conf)
		if err == nil {
			t.Fatalf("expected error because of inconsistent tensor len along dimension (2)")
		} else if err.Error() != "Of input data validation failed: expected data to have have equal length along every dimension" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Of([][][][]float64{
			{
				{{3., 3., 3.}},
				{{3., 3., 3.}},
				{{3., 3., 3.}},
			},
			{
				{{3., 3., 3.}},
				{{3., 3., 3.}},
			},
		}, conf)
		if err == nil {
			t.Fatalf("expected error because of inconsistent tensor len along dimension (2)")
		} else if err.Error() != "Of input data validation failed: expected data to have have equal length along every dimension" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		conf = &tensor.Config{Device: -1}

		/* ------------------------------ */

		_, err = tensor.Of([]float64{1}, conf)
		if err == nil {
			t.Fatalf("expected error because of invalid input device")
		} else if err.Error() != "Of tensor config data validation failed: invalid input device" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}

func TestValidationConcat(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		_, err := tensor.Concat(nil, 0)
		if err == nil {
			t.Fatalf("expected error because of the number of input tensors being less than (2)")
		} else if err.Error() != "Concat tensor implementation validation failed: expected at least (2) tensors: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Concat([]tensor.Tensor{nil}, 0)
		if err == nil {
			t.Fatalf("expected error because of the number of input tensors being less than (2)")
		} else if err.Error() != "Concat tensor implementation validation failed: expected at least (2) tensors: got (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Concat([]tensor.Tensor{nil, nil}, 0)
		if err == nil {
			t.Fatalf("expected error because of nil input tensors")
		} else if err.Error() != "Concat tensor implementation validation failed: unsupported tensor implementation" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err := tensor.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tensor.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = tensor.Concat([]tensor.Tensor{t1, t2}, 0)
		if err == nil {
			t.Fatalf("expected error because of having scalar tensors as input")
		} else if err.Error() != "Concat inputs' dimension validation failed: scalar tensor can not be concatenated: got tensor (0)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = tensor.Zeros([]int{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Zeros([]int{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t3, err := tensor.Zeros([]int{2, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 0)
		if err == nil {
			t.Fatalf("expected error because of the input tensors not having equal number of dimensions")
		} else if err.Error() != "Concat inputs' dimension validation failed: expected tensors to have the same number of dimensions: (2) != (1) for tensor (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = tensor.Zeros([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Zeros([]int{3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = tensor.Concat([]tensor.Tensor{t1, t2}, -1)
		if err == nil {
			t.Fatalf("expected error because of negative dimension")
		} else if err.Error() != "Concat inputs' dimension validation failed: expected concat dimension to be in range [0,1): got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Concat([]tensor.Tensor{t1, t2}, 1)
		if err == nil {
			t.Fatalf("expected error because of dimension (1) being out of range")
		} else if err.Error() != "Concat inputs' dimension validation failed: expected concat dimension to be in range [0,1): got (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = tensor.Zeros([]int{3, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Zeros([]int{3, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = tensor.Concat([]tensor.Tensor{t1, t2}, 2)
		if err == nil {
			t.Fatalf("expected error because of dimension (2) being out of range")
		} else if err.Error() != "Concat inputs' dimension validation failed: expected concat dimension to be in range [0,2): got (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = tensor.Zeros([]int{2, 2, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Zeros([]int{2, 2, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t3, err = tensor.Zeros([]int{3, 2, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 0)
		if err == nil {
			t.Fatalf("expected error because of size mismatch along dimension (2)")
		} else if err.Error() != "Concat inputs' dimension validation failed: expected tensor sizes to match in all dimensions except (0): (1) != (2) for dimension (2) for tensor (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = tensor.Zeros([]int{2, 1, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Zeros([]int{2, 2, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t3, err = tensor.Zeros([]int{3, 2, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 0)
		if err == nil {
			t.Fatalf("expected error because of size mismatch along dimension (1)")
		} else if err.Error() != "Concat inputs' dimension validation failed: expected tensor sizes to match in all dimensions except (0): (2) != (1) for dimension (1) for tensor (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = tensor.Zeros([]int{2, 1, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Zeros([]int{1, 2, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t3, err = tensor.Zeros([]int{2, 3, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 1)
		if err == nil {
			t.Fatalf("expected error because of size mismatch along dimension (0)")
		} else if err.Error() != "Concat inputs' dimension validation failed: expected tensor sizes to match in all dimensions except (1): (1) != (2) for dimension (0) for tensor (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}

func TestValidationAt(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tensor.Zeros([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.At()
		if err == nil {
			t.Fatalf("expected error because of incompatible index len (0) with dimension len (1)")
		} else if err.Error() != "At input index validation failed: expected index length to be equal to the number of dimensions: (0) != (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.At(0, 0)
		if err == nil {
			t.Fatalf("expected error because of incompatible index len (2) with dimension len (1)")
		} else if err.Error() != "At input index validation failed: expected index length to be equal to the number of dimensions: (2) != (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.At(-1)
		if err == nil {
			t.Fatalf("expected error because of negative index")
		} else if err.Error() != "At input index validation failed: expected index to be in range [0,1) at dimension (0): got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.At(1)
		if err == nil {
			t.Fatalf("expected error because of index (1) at dimension (0) being out of range [0,1)")
		} else if err.Error() != "At input index validation failed: expected index to be in range [0,1) at dimension (0): got (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = tensor.Zeros([]int{1, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.At(0)
		if err == nil {
			t.Fatalf("expected error because of incompatible index len (1) with dimension len (2)")
		} else if err.Error() != "At input index validation failed: expected index length to be equal to the number of dimensions: (1) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.At(0, 1, 0)
		if err == nil {
			t.Fatalf("expected error because of incompatible index len (3) with dimension len (2)")
		} else if err.Error() != "At input index validation failed: expected index length to be equal to the number of dimensions: (3) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.At(-2, -1)
		if err == nil {
			t.Fatalf("expected error because of negative index")
		} else if err.Error() != "At input index validation failed: expected index to be in range [0,1) at dimension (0): got (-2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.At(1, 0)
		if err == nil {
			t.Fatalf("expected error because of index (1) at dimension (0) being out of range [0,1)")
		} else if err.Error() != "At input index validation failed: expected index to be in range [0,1) at dimension (0): got (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.At(0, 2)
		if err == nil {
			t.Fatalf("expected error because of index (2) at dimension (1) being out of range [0,2)")
		} else if err.Error() != "At input index validation failed: expected index to be in range [0,2) at dimension (1): got (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}

func TestValidationSlice(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tensor.Of(2., conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Slice([]tensor.Range{{From: 0, To: 0}})
		if err == nil {
			t.Fatalf("expected error because of incompatible index len (1) with dimension len (0)")
		} else if err.Error() != "Slice input index validation failed: expected index length to be smaller than or equal to the number of dimensions: (1) > (0)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = tensor.Of([]float64{3.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Slice([]tensor.Range{{From: 0, To: 0}, {From: 0, To: 0}})
		if err == nil {
			t.Fatalf("expected error because of incompatible index len (2) with dimension len (1)")
		} else if err.Error() != "Slice input index validation failed: expected index length to be smaller than or equal to the number of dimensions: (2) > (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.Slice([]tensor.Range{{From: 1, To: 1}})
		if err == nil {
			t.Fatalf("expected error because of to index (0) not being larger than from index (0)")
		} else if err.Error() != "Slice input index validation failed: expected range 'From' to be smaller than 'To' except for special both (0) case (fetchAll): (1) >= (1) at dimension (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.Slice([]tensor.Range{{From: -1, To: 0}})
		if err == nil {
			t.Fatalf("expected error because of negative from index (-1)")
		} else if err.Error() != "Slice input index validation failed: expected index to be in range [0,1) at dimension (0): got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.Slice([]tensor.Range{{From: 1, To: 2}})
		if err == nil {
			t.Fatalf("expected error because of from index (1) being out of range [0,1) at dimension (0)")
		} else if err.Error() != "Slice input index validation failed: expected index to be in range [0,1) at dimension (0): got (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.Slice([]tensor.Range{{From: 0, To: 2}})
		if err == nil {
			t.Fatalf("expected error because of to index (2) being out of range [0,1) at dimension (0)")
		} else if err.Error() != "Slice input index validation failed: expected index to fall in range [0,1] at dimension (0): got [0,2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		ten, err = tensor.Of([]float64{1., 4.}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = ten.Slice([]tensor.Range{{From: 2, To: 3}})
		if err == nil {
			t.Fatalf("expected error because of from index (2) being out of range [0,2) at dimension (0)")
		} else if err.Error() != "Slice input index validation failed: expected index to be in range [0,2) at dimension (0): got (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = ten.Slice([]tensor.Range{{From: 1, To: 3}})
		if err == nil {
			t.Fatalf("expected error because of to index (3) being out of range [0,2) at dimension (0)")
		} else if err.Error() != "Slice input index validation failed: expected index to fall in range [0,2] at dimension (0): got [1,3)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}

func TestValidationPatch(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		t1, err := tensor.Zeros([]int{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Patch([]tensor.Range{}, nil)
		if err == nil {
			t.Fatalf("expected error because of nil input tensor")
		} else if err.Error() != fmt.Sprintf("Patch tensors' device validation failed: expected input tensor to be on %s", dev) {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = tensor.Zeros([]int{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err := tensor.Ones(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Patch([]tensor.Range{}, t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible number of dimensions")
		} else if err.Error() != "Patch input index or tensors' dimension validation failed: expected number of dimensions to match among source and target tensors: (0) != (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = tensor.Zeros([]int{1, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Ones([]int{1, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Patch([]tensor.Range{}, t2)
		if err == nil {
			t.Fatalf("expected error because of exceeding patch size at dimension (1)")
		} else if err.Error() != "Patch input index or tensors' dimension validation failed: expected source tensor size not to exceed that of target tensor at dimension (1): (2) > (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = tensor.Zeros([]int{3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Ones([]int{2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Patch([]tensor.Range{{From: 2, To: 4}}, t2)
		if err == nil {
			t.Fatalf("expected error because of incompatible index with target tensor")
		} else if err.Error() != "Patch input index or tensors' dimension validation failed: index incompatible with target tensor: expected index to fall in range [0,3] at dimension (0): got [2,4)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		t1, err = tensor.Zeros([]int{1, 3}, conf)
		if err != nil {
			t.Fatal(err)
		}

		t2, err = tensor.Ones([]int{1, 2}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = t1.Patch([]tensor.Range{{}, {From: 2, To: 3}}, t2)
		if err == nil {
			t.Fatalf("expected error because of index not covering source tensor at dimension (1)")
		} else if err.Error() != "Patch input index or tensors' dimension validation failed: expected index to exactly cover source tensor at dimension (1): #[2,3) != (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
