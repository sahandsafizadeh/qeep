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
