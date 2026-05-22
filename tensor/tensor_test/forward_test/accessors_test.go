package forward_test

import (
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
