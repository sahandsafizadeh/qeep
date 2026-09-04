package tensor_test

import (
	"slices"
	"sync"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestShape(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("Full(nil, 0) scalar tensor / Shape() / returns []", func(t *testing.T) {
			ten, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if shape := ten.Shape(); !slices.Equal(shape, []int{}) {
				t.Fatal("expected tensor to have shape [], got", shape)
			}
		})

		t.Run("Full([1], 0) 1-element 1D tensor / Shape() / returns [1]", func(t *testing.T) {
			ten, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if shape := ten.Shape(); !slices.Equal(shape, []int{1}) {
				t.Fatal("expected tensor to have shape [1], got", shape)
			}
		})

		t.Run("Full([2], 0) 1D tensor / Shape() / returns [2]", func(t *testing.T) {
			ten, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if shape := ten.Shape(); !slices.Equal(shape, []int{2}) {
				t.Fatal("expected tensor to have shape [2], got", shape)
			}
		})

		t.Run("Full([3,4], 0) 2D tensor / Shape() / returns [3,4]", func(t *testing.T) {
			ten, err := tensor.Full([]int{3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if shape := ten.Shape(); !slices.Equal(shape, []int{3, 4}) {
				t.Fatal("expected tensor to have shape [3, 4], got", shape)
			}
		})

		t.Run("Full([5,4,3,2,1], 0) 5D tensor / Shape() / returns [5,4,3,2,1]", func(t *testing.T) {
			ten, err := tensor.Full([]int{5, 4, 3, 2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if shape := ten.Shape(); !slices.Equal(shape, []int{5, 4, 3, 2, 1}) {
				t.Fatal("expected tensor to have shape [5, 4, 3, 2, 1], got", shape)
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("Full([5,4,3,2,1], 0) 5D tensor / concurrent repeated Shape() over every iteration / always returns [5,4,3,2,1]", func(t *testing.T) {
			const (
				ni = 1 << 4
				ng = 1 << 8
			)

			ten, err := tensor.Full([]int{5, 4, 3, 2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						if shape := ten.Shape(); !slices.Equal(shape, []int{5, 4, 3, 2, 1}) {
							t.Error("expected tensor to have shape [5, 4, 3, 2, 1], got", shape)
							return
						}
					}
				})
			}
			wg.Wait()
		})

		// ============================== side effects ==============================

		t.Run("Full([3,4], 0) 2D tensor / Shape() then mutate result / original shape unchanged", func(t *testing.T) {
			ten, err := tensor.Full([]int{3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			shape := ten.Shape()
			shape[0] = 99

			if shape := ten.Shape(); !slices.Equal(shape, []int{3, 4}) {
				t.Fatal("expected tensor shape to remain [3, 4] after mutating returned slice, got", shape)
			}
		})
	})
}

func TestNElems(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

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

		// ============================== extra functionalities ==============================

		t.Run("Full([5,4,3,2,1], 0) 5D tensor / concurrent repeated NElems() over every iteration / always returns 120", func(t *testing.T) {
			const (
				ni = 1 << 4
				ng = 1 << 8
			)

			ten, err := tensor.Full([]int{5, 4, 3, 2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						if nElems := ten.NElems(); nElems != 120 {
							t.Errorf("expected tensor to have (120) elements, got (%d)", nElems)
							return
						}
					}
				})
			}
			wg.Wait()
		})
	})
}

func TestSlice(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

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
				t.Fatal("expected tensors to be equal")
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
				t.Fatal("expected tensors to be equal")
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
				t.Fatal("expected tensors to be equal")
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
				t.Fatal("expected tensors to be equal")
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
				t.Fatal("expected tensors to be equal")
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
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor / Slice([1,4)) middle range / returns [2, 3, 4]", func(t *testing.T) {
			ten, err := tensor.Of([]float64{1., 2., 3., 4., 5.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 1, To: 4}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{2., 3., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
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
				t.Fatal("expected tensors to be equal")
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
				t.Fatal("expected tensors to be equal")
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
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor / Slice([1,3), [1,3)) center block / returns [[6, 7], [10, 11]]", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{1., 2., 3., 4.},
				{5., 6., 7., 8.},
				{9., 10., 11., 12.},
				{13., 14., 15., 16.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 1, To: 3}, {From: 1, To: 3}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{6., 7.},
				{10., 11.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
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
				t.Fatal("expected tensors to be equal")
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
				t.Fatal("expected tensors to be equal")
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
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[2] tensor / Slice([[0,1))) / Device() / returns the device the input was created on", func(t *testing.T) {
			ten, err := tensor.Of([]float64{1., 2.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 0, To: 1}})
			if err != nil {
				t.Fatal(err)
			}

			if d := act.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [2] tensor / Slice([[0,1))) / y is not gradient-tracked", func(t *testing.T) {
			ten, err := tensor.Of([]float64{1., 2.}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := ten.Slice([]tensor.Range{{From: 0, To: 1}})
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [2] tensor / Slice([[0,1))) / y is gradient-tracked", func(t *testing.T) {
			ten, err := tensor.Of([]float64{1., 2.}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := ten.Slice([]tensor.Range{{From: 0, To: 1}})
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor / Slice(nil) then BackPropagate / gradient is all-ones [3,4]", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Slice(nil)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [3,4] tensor / Slice([1:3]) then BackPropagate / gradient of x is 1 inside sliced rows, 0 outside", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Slice([]tensor.Range{{From: 1, To: 3}})
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Of([][]float64{
				{0., 0., 0., 0.},
				{1., 1., 1., 1.},
				{1., 1., 1., 1.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [4,5] tensor / Slice([1:4],[1:4]) then BackPropagate / gradient of x is 1 inside slice window, 0 outside", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{0., 1., 2., 3., 4.},
				{5., 6., 7., 8., 9.},
				{4., 3., 2., 1., 0.},
				{9., 8., 7., 6., 5.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
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

			act := x.Gradient()

			exp, err := tensor.Of([][]float64{
				{0., 0., 0., 0., 0.},
				{0., 1., 1., 1., 0.},
				{0., 1., 1., 1., 0.},
				{0., 1., 1., 1., 0.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [4,4,4] tensor / Slice([1:3],[1:3],[1:3]) then BackPropagate / gradient is 1 inside window, 0 outside", func(t *testing.T) {
			x, err := tensor.Full([]int{4, 4, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Slice([]tensor.Range{{From: 1, To: 3}, {From: 1, To: 3}, {From: 1, To: 3}})
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Of([][][]float64{
				{
					{0., 0., 0., 0.},
					{0., 0., 0., 0.},
					{0., 0., 0., 0.},
					{0., 0., 0., 0.},
				},
				{
					{0., 0., 0., 0.},
					{0., 1., 1., 0.},
					{0., 1., 1., 0.},
					{0., 0., 0., 0.},
				},
				{
					{0., 0., 0., 0.},
					{0., 1., 1., 0.},
					{0., 1., 1., 0.},
					{0., 0., 0., 0.},
				},
				{
					{0., 0., 0., 0.},
					{0., 0., 0., 0.},
					{0., 0., 0., 0.},
					{0., 0., 0., 0.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [2^20] 1D tensor / Slice([1,2^20-1)) / returns [2^20-2]", func(t *testing.T) {
			n := 1 << 20

			ten, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Slice([]tensor.Range{{From: 1, To: n - 1}})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{n - 2}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[2^10] 1D tensor / concurrent repeated Slice then Equals over every iteration / never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			ten, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{n - 2}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						act, err := ten.Slice([]tensor.Range{{From: 1, To: n - 1}})
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := act.Equals(exp); err != nil {
							t.Error(err)
							return
						} else if !eq {
							t.Error("expected tensors to be equal")
							return
						}
					}
				})
			}
			wg.Wait()
		})

		// ============================== side effects ==============================

		t.Run("1D tensor does not share ranges slice / Slice([0,1)) then mutate ranges / source and target tensors unchanged", func(t *testing.T) {
			ten, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			index := []tensor.Range{{From: 0, To: 1}}

			slc, err := ten.Slice(index)
			if err != nil {
				t.Fatal(err)
			}

			index[0] = tensor.Range{From: 1, To: 2}

			expTen, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expSlc, err := tensor.Of([]float64{1.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := ten.Equals(expTen); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := slc.Equals(expSlc); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
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
				t.Fatal("expected error because of incompatible index len (1) with dimension len (0)")
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
				t.Fatal("expected error because of incompatible index len (2) with dimension len (1)")
			} else if err.Error() != "Slice input index validation failed: expected index length to be smaller than or equal to the number of dimensions: (2) > (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("2D tensor / Slice with 3 ranges / returns error: index length exceeds dimensions", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{1., 2.},
				{3., 4.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Slice([]tensor.Range{{From: 0, To: 1}, {From: 0, To: 1}, {From: 0, To: 1}})
			if err == nil {
				t.Fatal("expected error because of incompatible index len (3) with dimension len (2)")
			} else if err.Error() != "Slice input index validation failed: expected index length to be smaller than or equal to the number of dimensions: (3) > (2)" {
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
				t.Fatal("expected error because of to index (0) not being larger than from index (0)")
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
				t.Fatal("expected error because of negative from index (-1)")
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
				t.Fatal("expected error because of from index (1) being out of range [0,1) at dimension (0)")
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
				t.Fatal("expected error because of to index (2) being out of range [0,1) at dimension (0)")
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
				t.Fatal("expected error because of from index (2) being out of range [0,2) at dimension (0)")
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
				t.Fatal("expected error because of to index (3) being out of range [0,2) at dimension (0)")
			} else if err.Error() != "Slice input index validation failed: expected index to fall in range [0,2] at dimension (0): got [1,3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("2D tensor / Slice([0,2),[0,3)) second range invalid / returns error: to index 3 out of range [0,2] at dimension 1", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{1., 2.},
				{3., 4.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Slice([]tensor.Range{{From: 0, To: 2}, {From: 0, To: 3}})
			if err == nil {
				t.Fatal("expected error because of to index (3) being out of range [0,2) at dimension (1)")
			} else if err.Error() != "Slice input index validation failed: expected index to fall in range [0,2] at dimension (1): got [0,3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("3D tensor / Slice([0,2),[3,4),[0,2)) second range invalid / returns error: from index 3 out of range [0,2) at dimension 1", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2.},
					{3., 4.},
				},
				{
					{5., 6.},
					{7., 8.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Slice([]tensor.Range{{From: 0, To: 2}, {From: 3, To: 4}, {From: 0, To: 2}})
			if err == nil {
				t.Fatal("expected error because of from index (3) being out of range [0,2) at dimension (1)")
			} else if err.Error() != "Slice input index validation failed: expected index to be in range [0,2) at dimension (1): got (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("3D tensor / Slice([0,2),[0,2),[1,5)) third range invalid / returns error: to index 5 out of range [0,2] at dimension 2", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2.},
					{3., 4.},
				},
				{
					{5., 6.},
					{7., 8.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Slice([]tensor.Range{{From: 0, To: 2}, {From: 0, To: 2}, {From: 1, To: 5}})
			if err == nil {
				t.Fatal("expected error because of to index (5) being out of range [0,2) at dimension (2)")
			} else if err.Error() != "Slice input index validation failed: expected index to fall in range [0,2] at dimension (2): got [1,5)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
