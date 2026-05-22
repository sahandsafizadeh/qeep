// Accessors are the second group of functions under test in this package.
// They are verified using only the primary functions (Full, Of, At, Equals)
// established as a baseline in primary_ops_test.go.
package forward_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestNElems(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 0) scalar tensor / NElems() / returns 1", func(t *testing.T) {
			ten, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if nElems := ten.NElems(); nElems != 1 {
				t.Fatalf("expected tensor to have (1) element, got (%d)", nElems)
			}
		})

		t.Run("Full([1], 0) 1-element 1D tensor / NElems() / returns 1", func(t *testing.T) {
			ten, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if nElems := ten.NElems(); nElems != 1 {
				t.Fatalf("expected tensor to have (1) element, got (%d)", nElems)
			}
		})

		t.Run("Full([2], 0) 1D tensor / NElems() / returns 2", func(t *testing.T) {
			ten, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if nElems := ten.NElems(); nElems != 2 {
				t.Fatalf("expected tensor to have (2) elements, got (%d)", nElems)
			}
		})

		t.Run("Full([3,4], 0) 2D tensor / NElems() / returns 12", func(t *testing.T) {
			ten, err := tensor.Full([]int{3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if nElems := ten.NElems(); nElems != 12 {
				t.Fatalf("expected tensor to have (12) elements, got (%d)", nElems)
			}
		})

		t.Run("Full([5,4,3,2,1], 0) 5D tensor / NElems() / returns 120", func(t *testing.T) {
			ten, err := tensor.Full([]int{5, 4, 3, 2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if nElems := ten.NElems(); nElems != 120 {
				t.Fatalf("expected tensor to have (120) elements, got (%d)", nElems)
			}
		})
	})
}

func TestShape(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 0) scalar tensor / Shape() / returns []", func(t *testing.T) {
			ten, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if shape := ten.Shape(); !slices.Equal(shape, []int{}) {
				t.Fatalf("expected tensor to have shape [], got %v", shape)
			}
		})

		t.Run("Full([1], 0) 1-element 1D tensor / Shape() / returns [1]", func(t *testing.T) {
			ten, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if shape := ten.Shape(); !slices.Equal(shape, []int{1}) {
				t.Fatalf("expected tensor to have shape [1], got %v", shape)
			}
		})

		t.Run("Full([2], 0) 1D tensor / Shape() / returns [2]", func(t *testing.T) {
			ten, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if shape := ten.Shape(); !slices.Equal(shape, []int{2}) {
				t.Fatalf("expected tensor to have shape [2], got %v", shape)
			}
		})

		t.Run("Full([3,4], 0) 2D tensor / Shape() / returns [3,4]", func(t *testing.T) {
			ten, err := tensor.Full([]int{3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if shape := ten.Shape(); !slices.Equal(shape, []int{3, 4}) {
				t.Fatalf("expected tensor to have shape [3, 4], got %v", shape)
			}
		})

		t.Run("Full([5,4,3,2,1], 0) 5D tensor / Shape() / returns [5,4,3,2,1]", func(t *testing.T) {
			ten, err := tensor.Full([]int{5, 4, 3, 2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if shape := ten.Shape(); !slices.Equal(shape, []int{5, 4, 3, 2, 1}) {
				t.Fatalf("expected tensor to have shape [5, 4, 3, 2, 1], got %v", shape)
			}
		})
	})
}

func TestSlice(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("scalar tensor / Slice(nil) / returns same scalar", func(t *testing.T) {
			ten, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice(nil)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("1D tensor / Slice([0,1)) / returns [3]", func(t *testing.T) {
			ten, err := tensor.Of([]float64{3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 0, To: 1}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("1D tensor / Slice([{}]) fetchAll / returns [4]", func(t *testing.T) {
			ten, err := tensor.Of([]float64{4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("1D tensor / Slice([0,1)) / returns [1]", func(t *testing.T) {
			ten, err := tensor.Of([]float64{1., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 0, To: 1}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{1.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("1D tensor / Slice([1,2)) / returns [4]", func(t *testing.T) {
			ten, err := tensor.Of([]float64{1., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 1, To: 2}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("1D tensor / Slice([0,2)) / returns [1, 4]", func(t *testing.T) {
			ten, err := tensor.Of([]float64{1., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 0, To: 2}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{1., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("2D tensor / Slice([0,1)) first row / returns [[-1]]", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{{-1.}, {-2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 0, To: 1}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{{-1.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("2D tensor / Slice([1,2)) second row / returns [[-2]]", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{{-1.}, {-2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 1, To: 2}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{{-2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("2D tensor / Slice([{}, [0,1))) all rows col 0 / returns [[-1], [-2]]", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{{-1.}, {-2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{}, {From: 0, To: 1}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{{-1.}, {-2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("3x3x3 tensor / Slice([{}, [1,2))) / returns middle rows across all batches", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{}, {From: 1, To: 2}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{{2., 4., 6.}},
				{{9., 7., 5.}},
				{{1., 2., 6.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("3x3x3 tensor / Slice([[0,2), {}, [1,3))) first 2 batches cols 1-2 / returns subtensor", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 0, To: 2}, {}, {From: 1, To: 3}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("1x2x3x4 tensor / Slice([{}, [1,2), [1,3))) / returns subtensor", func(t *testing.T) {
			ten, err := tensor.Of([][][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{}, {From: 1, To: 2}, {From: 1, To: 3}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][][]float64{
				{
					{
						{1., 2., 3., 4.},
						{1., 2., 3., 4.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		// ============================== validations ==============================

		t.Run("scalar tensor / Slice with 1 range / returns error: index length exceeds dimensions", func(t *testing.T) {
			ten, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Slice([]tensor.Range{{From: 0, To: 0}})
			if err == nil {
				t.Fatalf("expected error because of incompatible index len (1) with dimension len (0)")
			} else if err.Error() != "Slice input index validation failed: expected index length to be smaller than or equal to the number of dimensions: (1) > (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("1D tensor / Slice with 2 ranges / returns error: index length exceeds dimensions", func(t *testing.T) {
			ten, err := tensor.Of([]float64{3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Slice([]tensor.Range{{From: 0, To: 0}, {From: 0, To: 0}})
			if err == nil {
				t.Fatalf("expected error because of incompatible index len (2) with dimension len (1)")
			} else if err.Error() != "Slice input index validation failed: expected index length to be smaller than or equal to the number of dimensions: (2) > (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("1D tensor / Slice([1,1)) / returns error: from not smaller than to", func(t *testing.T) {
			ten, err := tensor.Of([]float64{3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Slice([]tensor.Range{{From: 1, To: 1}})
			if err == nil {
				t.Fatalf("expected error because of to index (0) not being larger than from index (0)")
			} else if err.Error() != "Slice input index validation failed: expected range 'From' to be smaller than 'To' except for special both (0) case (fetchAll): (1) >= (1) at dimension (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("1D tensor / Slice([-1,0)) / returns error: negative from index at dimension 0", func(t *testing.T) {
			ten, err := tensor.Of([]float64{3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Slice([]tensor.Range{{From: -1, To: 0}})
			if err == nil {
				t.Fatalf("expected error because of negative from index (-1)")
			} else if err.Error() != "Slice input index validation failed: expected index to be in range [0,1) at dimension (0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("1D tensor / Slice([1,2)) / returns error: from index 1 out of range [0,1)", func(t *testing.T) {
			ten, err := tensor.Of([]float64{3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Slice([]tensor.Range{{From: 1, To: 2}})
			if err == nil {
				t.Fatalf("expected error because of from index (1) being out of range [0,1) at dimension (0)")
			} else if err.Error() != "Slice input index validation failed: expected index to be in range [0,1) at dimension (0): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("1D tensor / Slice([0,2)) / returns error: to index 2 out of range [0,1]", func(t *testing.T) {
			ten, err := tensor.Of([]float64{3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Slice([]tensor.Range{{From: 0, To: 2}})
			if err == nil {
				t.Fatalf("expected error because of to index (2) being out of range [0,1) at dimension (0)")
			} else if err.Error() != "Slice input index validation failed: expected index to fall in range [0,1] at dimension (0): got [0,2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("1D tensor / Slice([2,3)) / returns error: from index 2 out of range [0,2)", func(t *testing.T) {
			ten, err := tensor.Of([]float64{1., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Slice([]tensor.Range{{From: 2, To: 3}})
			if err == nil {
				t.Fatalf("expected error because of from index (2) being out of range [0,2) at dimension (0)")
			} else if err.Error() != "Slice input index validation failed: expected index to be in range [0,2) at dimension (0): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("1D tensor / Slice([1,3)) / returns error: to index 3 out of range [0,2]", func(t *testing.T) {
			ten, err := tensor.Of([]float64{1., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Slice([]tensor.Range{{From: 1, To: 3}})
			if err == nil {
				t.Fatalf("expected error because of to index (3) being out of range [0,2) at dimension (0)")
			} else if err.Error() != "Slice input index validation failed: expected index to fall in range [0,2] at dimension (0): got [1,3)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestPatch(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 0.) scalar and Full(nil, 1.) scalar / Patch(nil) / replaces scalar value", func(t *testing.T) {
			t1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := t1.Patch(nil, t2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Full([2], 0.) and Full([1], 1.) / Patch([{1,2}]) / patches last element", func(t *testing.T) {
			t1, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := t1.Patch([]tensor.Range{{From: 1, To: 2}}, t2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{0., 1.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Full([3,3], 0.) and Full([2,2], 1.) / Patch([{0,2},{0,2}]) / patches top-left 2x2 block", func(t *testing.T) {
			t1, err := tensor.Full([]int{3, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := t1.Patch([]tensor.Range{{From: 0, To: 2}, {From: 0, To: 2}}, t2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{1., 1., 0.},
				{1., 1., 0.},
				{0., 0., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Full([3,3], 0.) and Full([2,2], 1.) / Patch([{0,2},{1,3}]) / patches top-right 2x2 block", func(t *testing.T) {
			t1, err := tensor.Full([]int{3, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := t1.Patch([]tensor.Range{{From: 0, To: 2}, {From: 1, To: 3}}, t2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{0., 1., 1.},
				{0., 1., 1.},
				{0., 0., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Full([3,3], 0.) and Full([2,2], 1.) / Patch([{1,3},{0,2}]) / patches bottom-left 2x2 block", func(t *testing.T) {
			t1, err := tensor.Full([]int{3, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := t1.Patch([]tensor.Range{{From: 1, To: 3}, {From: 0, To: 2}}, t2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{0., 0., 0.},
				{1., 1., 0.},
				{1., 1., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Full([3,3], 0.) and Full([2,2], 1.) / Patch([{1,3},{1,3}]) / patches bottom-right 2x2 block", func(t *testing.T) {
			t1, err := tensor.Full([]int{3, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := t1.Patch([]tensor.Range{{From: 1, To: 3}, {From: 1, To: 3}}, t2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{0., 0., 0.},
				{0., 1., 1.},
				{0., 1., 1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Full([4,3,2], 0.) and Full([3,2,1], 1.) / Patch(nil) / patches from origin with nil ranges", func(t *testing.T) {
			t1, err := tensor.Full([]int{4, 3, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{3, 2, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := t1.Patch(nil, t2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Full([4,3,2], 0.) and Full([3,2,1], 1.) / Patch([{1,4}]) / patches from dim-0 offset 1", func(t *testing.T) {
			t1, err := tensor.Full([]int{4, 3, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{3, 2, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := t1.Patch([]tensor.Range{{From: 1, To: 4}}, t2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Full([4,3,2], 0.) and Full([3,2,1], 1.) / Patch([{1,4},{1,3},{1,2}]) / patches from all-dimension offset", func(t *testing.T) {
			t1, err := tensor.Full([]int{4, 3, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{3, 2, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := t1.Patch([]tensor.Range{{From: 1, To: 4}, {From: 1, To: 3}, {From: 1, To: 2}}, t2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Of([[1,2],[3,4]]) tensor / Patch(nil, self) / returns same tensor", func(t *testing.T) {
			t1, err := tensor.Of([][]float64{
				{1., 2.},
				{3., 4.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := t1.Patch(nil, t1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{1., 2.},
				{3., 4.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		// ============================== validations ==============================

		t.Run("Full([2], 0.) / Patch([], nil) / returns error: source not on device", func(t *testing.T) {
			t1, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = t1.Patch([]tensor.Range{}, nil)
			if err == nil {
				t.Fatalf("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Patch tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([2], 0.) and Full(nil, 0.) scalar / Patch([]) / returns error: dimension count mismatch", func(t *testing.T) {
			t1, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = t1.Patch([]tensor.Range{}, t2)
			if err == nil {
				t.Fatalf("expected error because of incompatible number of dimensions")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: expected number of dimensions to match among source and target tensors: (0) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1,1], 0.) and Full([1,2], 0.) / Patch([]) / returns error: source exceeds target at dim 1", func(t *testing.T) {
			t1, err := tensor.Full([]int{1, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = t1.Patch([]tensor.Range{}, t2)
			if err == nil {
				t.Fatalf("expected error because of exceeding patch size at dimension (1)")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: expected source tensor size not to exceed that of target tensor at dimension (1): (2) > (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([3], 0.) and Full([2], 0.) / Patch([{2,4}]) / returns error: index out of target range", func(t *testing.T) {
			t1, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = t1.Patch([]tensor.Range{{From: 2, To: 4}}, t2)
			if err == nil {
				t.Fatalf("expected error because of incompatible index with target tensor")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: index incompatible with target tensor: expected index to fall in range [0,3] at dimension (0): got [2,4)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1,3], 0.) and Full([1,2], 0.) / Patch([{},{2,3}]) / returns error: index does not cover source at dim 1", func(t *testing.T) {
			t1, err := tensor.Full([]int{1, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = t1.Patch([]tensor.Range{{}, {From: 2, To: 3}}, t2)
			if err == nil {
				t.Fatalf("expected error because of index not covering source tensor at dimension (1)")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: expected index to exactly cover source tensor at dimension (1): #[2,3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
