package forward_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestEye(t *testing.T) {

	// ============================== main paths ==============================

	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {
		t.Run("Eye(1) 1x1 identity matrix / Equals / returns true", func(t *testing.T) {
			act, err := tensor.Eye(1, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{{1.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Eye(5) 5x5 identity matrix / Equals / returns true", func(t *testing.T) {
			act, err := tensor.Eye(5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{1., 0., 0., 0., 0.},
				{0., 1., 0., 0., 0.},
				{0., 0., 1., 0., 0.},
				{0., 0., 0., 1., 0.},
				{0., 0., 0., 0., 1.},
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

		t.Run("Eye(-1) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Eye(-1, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of non-positive dimension")
			} else if err.Error() != "Eye input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Eye(0) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Eye(0, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of non-positive dimension")
			} else if err.Error() != "Eye input dimension validation failed: expected positive dimension sizes: got (0) at position (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Eye(1) with invalid device / returns error: invalid device", func(t *testing.T) {
			_, err := tensor.Eye(1, &tensor.Config{Device: -1})
			if err == nil {
				t.Fatalf("expected error because of invalid input device")
			} else if err.Error() != "Eye tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestRandU(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== side effects ==============================

		t.Run("RandU([3,4], -1, 1) does not share dims slice / Shape() after mutating dims / returns [3,4]", func(t *testing.T) {
			dims := []int{3, 4}
			ten, err := tensor.RandU(dims, -1., 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			dims[0] = 1
			dims[1] = 1
			shape := ten.Shape()
			if !slices.Equal(shape, []int{3, 4}) {
				t.Fatalf("expected tensor to have shape [3, 4], got %v", shape)
			}
		})

		// ============================== validations ==============================

		t.Run("RandU(nil, 0, -1) / lower bound >= upper bound / returns error: lower bound not less than upper bound", func(t *testing.T) {
			_, err := tensor.RandU(nil, 0., -1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of lower bound not being less than upper bound")
			} else if err.Error() != "RandU random parameter validation failed: expected uniform random lower bound to be less than the upper bound: (0.000000) >= (-1.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandU(nil, 1, 1) / equal bounds / returns error: lower bound not less than upper bound", func(t *testing.T) {
			_, err := tensor.RandU(nil, 1., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of lower bound not being less than upper bound")
			} else if err.Error() != "RandU random parameter validation failed: expected uniform random lower bound to be less than the upper bound: (1.000000) >= (1.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandU([-1], -1, 1) / negative dimension / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.RandU([]int{-1}, -1., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of non-positive dimension")
			} else if err.Error() != "RandU input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandU([1x7], -1, 1) / too many dimensions / returns error: exceeds max dimensions", func(t *testing.T) {
			_, err := tensor.RandU([]int{1, 1, 1, 1, 1, 1, 1}, -1., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of too many dimensions")
			} else if err.Error() != "RandU input dimension validation failed: expected at most (6) dimensions: got (7)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandU(nil, 0, 1) with invalid device / returns error: invalid input device", func(t *testing.T) {
			_, err := tensor.RandU(nil, 0., 1., &tensor.Config{Device: -1})
			if err == nil {
				t.Fatalf("expected error because of invalid input device")
			} else if err.Error() != "RandU tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})

	})
}

func TestRandN(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== side effects ==============================

		t.Run("RandN([3,4], 0, 1) does not share dims slice / Shape() after mutating dims / returns [3,4]", func(t *testing.T) {
			dims := []int{3, 4}
			ten, err := tensor.RandN(dims, 0., 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			dims[0] = 1
			dims[1] = 1
			shape := ten.Shape()
			if !slices.Equal(shape, []int{3, 4}) {
				t.Fatalf("expected tensor to have shape [3, 4], got %v", shape)
			}
		})

		// ============================== validations ==============================

		t.Run("RandN(nil, 0, -1) / negative standard deviation / returns error: std dev not positive", func(t *testing.T) {
			_, err := tensor.RandN(nil, 0., -1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of non-positive standard deviation")
			} else if err.Error() != "RandN random parameter validation failed: expected normal random standard deviation to be positive: got (-1.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandN(nil, -1, 0) / zero standard deviation / returns error: std dev not positive", func(t *testing.T) {
			_, err := tensor.RandN(nil, -1., 0., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of non-positive standard deviation")
			} else if err.Error() != "RandN random parameter validation failed: expected normal random standard deviation to be positive: got (0.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandN([-1], 0, 1) / negative dimension / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.RandN([]int{-1}, 0., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of non-positive dimension")
			} else if err.Error() != "RandN input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandN([1x9], 0, 1) / too many dimensions / returns error: exceeds max dimensions", func(t *testing.T) {
			_, err := tensor.RandN([]int{1, 1, 1, 1, 1, 1, 1, 1, 1}, 0., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of too many dimensions")
			} else if err.Error() != "RandN input dimension validation failed: expected at most (6) dimensions: got (9)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandN(nil, 0, 1) with invalid device / returns error: invalid input device", func(t *testing.T) {
			_, err := tensor.RandN(nil, 0., 1., &tensor.Config{Device: -1})
			if err == nil {
				t.Fatalf("expected error because of invalid input device")
			} else if err.Error() != "RandN tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})

	})
}

func TestOfSlice(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Of(2.0) scalar / Slice(nil) / returns same scalar", func(t *testing.T) {
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

		t.Run("Of([3.0]) 1D tensor / Slice([0,1)) / returns [3.0]", func(t *testing.T) {
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

		t.Run("Of([4.0]) 1D tensor / Slice([{}]) fetchAll / returns [4.0]", func(t *testing.T) {
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

		t.Run("Of([1.0, 4.0]) 1D tensor / Slice([0,1)) / returns [1.0]", func(t *testing.T) {
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

		t.Run("Of([1.0, 4.0]) 1D tensor / Slice([1,2)) / returns [4.0]", func(t *testing.T) {
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

		t.Run("Of([1.0, 4.0]) 1D tensor / Slice([0,2)) / returns [1.0, 4.0]", func(t *testing.T) {
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

		t.Run("Of([[-1.0], [-2.0]]) 2D tensor / Slice([0,1)) first row / returns [[-1.0]]", func(t *testing.T) {
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

		t.Run("Of([[-1.0], [-2.0]]) 2D tensor / Slice([1,2)) second row / returns [[-2.0]]", func(t *testing.T) {
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

		t.Run("Of([[-1.0], [-2.0]]) 2D tensor / Slice([{}, [0,1))) all rows col 0 / returns [[-1.0], [-2.0]]", func(t *testing.T) {
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

		t.Run("Of(3x3x3 tensor) / Slice([{}, [1,2))) / returns middle rows across all batches", func(t *testing.T) {
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

		t.Run("Of(3x3x3 tensor) / Slice([[0,2), {}, [1,3))) first 2 batches cols 1-2 / returns subtensor", func(t *testing.T) {
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

		t.Run("Of(1x2x3x4 tensor) / Slice([{}, [1,2), [1,3))) / returns subtensor", func(t *testing.T) {
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

		// ============================== side effects ==============================

		t.Run("Of([5.0]) does not share 1D input slice / Slice(nil) after mutating source / returns 5.0", func(t *testing.T) {
			d1 := []float64{5.}

			ten, err := tensor.Of(d1, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			d1[0] = 3.

			act, err := ten.Slice(nil)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{5.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Of([[[[5.0]]]]) does not share 4D input slice / Slice(nil) after mutating source / returns 5.0", func(t *testing.T) {
			d4 := [][][][]float64{{{{5.}}}}

			ten, err := tensor.Of(d4, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			d4[0][0][0][0] = 3.

			act, err := ten.Slice(nil)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][][]float64{{{{5.}}}}, &tensor.Config{Device: dev})
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

		t.Run("Of([]float64{}) / returns error: zero length along dimension", func(t *testing.T) {
			_, err := tensor.Of([]float64{}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of zero len along dimension (0)")
			} else if err.Error() != "Of input data validation failed: expected data to not have zero length along any dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][]float64{}) / returns error: zero length along dimension", func(t *testing.T) {
			_, err := tensor.Of([][]float64{}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of zero len along dimension (0)")
			} else if err.Error() != "Of input data validation failed: expected data to not have zero length along any dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][][]float64{}) / returns error: zero length along dimension", func(t *testing.T) {
			_, err := tensor.Of([][][]float64{}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of zero len along dimension (0)")
			} else if err.Error() != "Of input data validation failed: expected data to not have zero length along any dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][][][]float64{}) / returns error: zero length along dimension", func(t *testing.T) {
			_, err := tensor.Of([][][][]float64{}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of zero len along dimension (0)")
			} else if err.Error() != "Of input data validation failed: expected data to not have zero length along any dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][]float64{{}, {}}) / returns error: zero length along inner dimension", func(t *testing.T) {
			_, err := tensor.Of([][]float64{{}, {}}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of zero len along dimension (1)")
			} else if err.Error() != "Of input data validation failed: expected data to not have zero length along any dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][]float64{{}, {-1.}}) / returns error: zero length along inner dimension", func(t *testing.T) {
			_, err := tensor.Of([][]float64{{}, {-1.}}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of zero len along dimension (1)")
			} else if err.Error() != "Of input data validation failed: expected data to not have zero length along any dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][][]float64{{{}}}) / returns error: zero length along inner dimension", func(t *testing.T) {
			_, err := tensor.Of([][][]float64{{{}}}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of zero len along dimension (1)")
			} else if err.Error() != "Of input data validation failed: expected data to not have zero length along any dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of(3x3x3 with inconsistent last row) / returns error: unequal lengths along dimension", func(t *testing.T) {
			_, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of inconsistent tensor len along dimension (2)")
			} else if err.Error() != "Of input data validation failed: expected data to have have equal length along every dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of(1x3x3x3 with inconsistent inner sub-tensor) / returns error: unequal lengths along dimension", func(t *testing.T) {
			_, err := tensor.Of([][][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of inconsistent tensor len along dimension (2)")
			} else if err.Error() != "Of input data validation failed: expected data to have have equal length along every dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of(2x3x1x3 with inconsistent outer batch) / returns error: unequal lengths along dimension", func(t *testing.T) {
			_, err := tensor.Of([][][][]float64{
				{
					{{3., 3., 3.}},
					{{3., 3., 3.}},
					{{3., 3., 3.}},
				},
				{
					{{3., 3., 3.}},
					{{3., 3., 3.}},
				},
			}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of inconsistent tensor len along dimension (2)")
			} else if err.Error() != "Of input data validation failed: expected data to have have equal length along every dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([1.0]) with invalid device / returns error: invalid device", func(t *testing.T) {
			_, err := tensor.Of([]float64{1}, &tensor.Config{Device: -1})
			if err == nil {
				t.Fatalf("expected error because of invalid input device")
			} else if err.Error() != "Of tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of(2.0) scalar tensor / Slice with 1 range / returns error: index length exceeds dimensions", func(t *testing.T) {
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

		t.Run("Of([3.0]) 1D tensor / Slice with 2 ranges / returns error: index length exceeds dimensions", func(t *testing.T) {
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

		t.Run("Of([3.0]) 1D tensor / Slice([1,1)) / returns error: from not smaller than to", func(t *testing.T) {
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

		t.Run("Of([3.0]) 1D tensor / Slice([-1,0)) / returns error: negative from index at dimension 0", func(t *testing.T) {
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

		t.Run("Of([3.0]) 1D tensor / Slice([1,2)) / returns error: from index 1 out of range [0,1)", func(t *testing.T) {
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

		t.Run("Of([3.0]) 1D tensor / Slice([0,2)) / returns error: to index 2 out of range [0,1]", func(t *testing.T) {
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

		t.Run("Of([1.0, 4.0]) 1D tensor / Slice([2,3)) / returns error: from index 2 out of range [0,2)", func(t *testing.T) {
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

		t.Run("Of([1.0, 4.0]) 1D tensor / Slice([1,3)) / returns error: to index 3 out of range [0,2]", func(t *testing.T) {
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

func TestDevice(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		ten, err := tensor.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		if d := ten.Device(); d != dev {
			t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
		}

		/* ------------------------------ */

	})
}

func TestValidationFullZerosOnes(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// TODO: don't forget to add the skipped scenarios from full

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		_, err := tensor.Zeros([]int{2, 0, 1}, conf)
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

		_, err = tensor.Zeros([]int{1, 1, 1, 1, 1, 1, 1}, conf)
		if err == nil {
			t.Fatalf("expected error because of too many dimensions")
		} else if err.Error() != "Zeros input dimension validation failed: expected at most (6) dimensions: got (7)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = tensor.Ones([]int{1, 1, 1, 1, 1, 1, 1, 1}, conf)
		if err == nil {
			t.Fatalf("expected error because of too many dimensions")
		} else if err.Error() != "Ones input dimension validation failed: expected at most (6) dimensions: got (8)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		conf = &tensor.Config{Device: -1}

		/* ------------------------------ */

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

		_, err = tensor.RandU([]int{1, 1, 1, 1, 1, 1, 1}, -1., 1., conf)
		if err == nil {
			t.Fatalf("expected error because of too many dimensions")
		} else if err.Error() != "RandU input dimension validation failed: expected at most (6) dimensions: got (7)" {
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

		_, err = tensor.RandN([]int{1, 1, 1, 1, 1, 1, 1, 1, 1}, 0., 1., conf)
		if err == nil {
			t.Fatalf("expected error because of too many dimensions")
		} else if err.Error() != "RandN input dimension validation failed: expected at most (6) dimensions: got (9)" {
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
