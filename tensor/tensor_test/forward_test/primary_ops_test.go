// Package forward_test verifies the forward pass (non-gradient) behavior of all tensor operations.
//
// Some initializers, accessors and operators are tested together rather than in isolation
// to resolve a chicken-and-egg dependency: verifying that a tensor was initialized correctly
// requires reading its values back, and trusting that a read is correct requires knowing the
// tensor was created correctly. By cross-validating both in the same suite, they establish a
// mutually consistent baseline that all other tests in this package build upon.
//
// Full, At, Of, and Equals are the primary functions under test here, as they underpin this
// baseline and are relied upon throughout the rest of the test suite.
package forward_test

import (
	"fmt"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestFullAt(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, -1) scalar tensor / At() with no indices / returns -1", func(t *testing.T) {
			ten, err := tensor.Full(nil, -1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := ten.At(); err != nil {
				t.Fatal(err)
			} else if int(val) != -1 {
				t.Fatalf("expected (-1) as scalar tensor value, got (%f)", val)
			}
		})

		t.Run("Full([1], 9) 1D tensor / At(0) / returns 9", func(t *testing.T) {
			ten, err := tensor.Full([]int{1}, 9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := ten.At(0); err != nil {
				t.Fatal(err)
			} else if int(val) != 9 {
				t.Fatalf("expected (9) as tensor value in position [0], got (%f)", val)
			}
		})

		t.Run("Full([1,2], 0) 2D tensor / At(i,j) for all positions / returns 0", func(t *testing.T) {
			ten, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
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
		})

		t.Run("Full([4,3,2,1], 5) 4D tensor / At(i,j,k,u) for all positions / returns 5", func(t *testing.T) {
			ten, err := tensor.Full([]int{4, 3, 2, 1}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			for i := range 4 {
				for j := range 3 {
					for k := range 2 {
						for u := range 1 {
							if val, err := ten.At(i, j, k, u); err != nil {
								t.Fatal(err)
							} else if int(val) != 5 {
								t.Fatalf("expected (5) as tensor value in position [%d,%d,%d,%d], got (%f)", i, j, k, u, val)
							}
						}
					}
				}
			}
		})

		// ============================== side effects ==============================

		t.Run("Full([3,4], 1) does not share dims slice / At(2,3) after mutating dims / returns 1", func(t *testing.T) {
			dims := []int{3, 4}

			ten, err := tensor.Full(dims, 1., &tensor.Config{Device: dev})
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
		})

		// ============================== validations ==============================

		t.Run("Full([-1]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Full([]int{-1}, 2., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: Full input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([0]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Full([]int{0}, 2., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: Full input dimension validation failed: expected positive dimension sizes: got (0) at position (0)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1,-2]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Full([]int{1, -2}, 2., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: Full input dimension validation failed: expected positive dimension sizes: got (-2) at position (1)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([2,0,1]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Full([]int{2, 0, 1}, 2., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: Full input dimension validation failed: expected positive dimension sizes: got (0) at position (1)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1,1,1,1,1,1,1]) / returns error: too many dimensions", func(t *testing.T) {
			_, err := tensor.Full([]int{1, 1, 1, 1, 1, 1, 1}, 2., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of too many dimensions")
			} else if err.Error() != fmt.Sprintf("%s initialization: Full input dimension validation failed: expected at most (6) dimensions: got (7)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full(nil) with invalid device / returns error: invalid device", func(t *testing.T) {
			_, err := tensor.Full(nil, 2., &tensor.Config{Device: -1})
			if err == nil {
				t.Fatal("expected error because of invalid input device")
			} else if err.Error() != "Full tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1]) 1D tensor / At() with wrong index count / returns error: index length 0 != dimensions 1", func(t *testing.T) {
			ten, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.At()
			if err == nil {
				t.Fatal("expected error because of incompatible index len (0) with dimension len (1)")
			} else if err.Error() != "At input index validation failed: expected index length to be equal to the number of dimensions: (0) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1]) 1D tensor / At(0,0) with wrong index count / returns error: index length 2 != dimensions 1", func(t *testing.T) {
			ten, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.At(0, 0)
			if err == nil {
				t.Fatal("expected error because of incompatible index len (2) with dimension len (1)")
			} else if err.Error() != "At input index validation failed: expected index length to be equal to the number of dimensions: (2) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1]) 1D tensor / At(-1) / returns error: negative index at dimension 0", func(t *testing.T) {
			ten, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.At(-1)
			if err == nil {
				t.Fatal("expected error because of negative index")
			} else if err.Error() != "At input index validation failed: expected index to be in range [0,1) at dimension (0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1]) 1D tensor / At(1) / returns error: index out of range [0,1) at dimension 0", func(t *testing.T) {
			ten, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.At(1)
			if err == nil {
				t.Fatal("expected error because of index (1) at dimension (0) being out of range [0,1)")
			} else if err.Error() != "At input index validation failed: expected index to be in range [0,1) at dimension (0): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1,2]) 2D tensor / At(0) with wrong index count / returns error: index length 1 != dimensions 2", func(t *testing.T) {
			ten, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.At(0)
			if err == nil {
				t.Fatal("expected error because of incompatible index len (1) with dimension len (2)")
			} else if err.Error() != "At input index validation failed: expected index length to be equal to the number of dimensions: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1,2]) 2D tensor / At(0,1,0) with wrong index count / returns error: index length 3 != dimensions 2", func(t *testing.T) {
			ten, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.At(0, 1, 0)
			if err == nil {
				t.Fatal("expected error because of incompatible index len (3) with dimension len (2)")
			} else if err.Error() != "At input index validation failed: expected index length to be equal to the number of dimensions: (3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1,2]) 2D tensor / At(-2,-1) / returns error: negative index at dimension 0", func(t *testing.T) {
			ten, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.At(-2, -1)
			if err == nil {
				t.Fatal("expected error because of negative index")
			} else if err.Error() != "At input index validation failed: expected index to be in range [0,1) at dimension (0): got (-2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1,2]) 2D tensor / At(1,0) / returns error: index 1 out of range [0,1) at dimension 0", func(t *testing.T) {
			ten, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.At(1, 0)
			if err == nil {
				t.Fatal("expected error because of index (1) at dimension (0) being out of range [0,1)")
			} else if err.Error() != "At input index validation failed: expected index to be in range [0,1) at dimension (0): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1,2]) 2D tensor / At(0,2) / returns error: index 2 out of range [0,2) at dimension 1", func(t *testing.T) {
			ten, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.At(0, 2)
			if err == nil {
				t.Fatal("expected error because of index (2) at dimension (1) being out of range [0,2)")
			} else if err.Error() != "At input index validation failed: expected index to be in range [0,2) at dimension (1): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestOfAt(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Of(2) scalar / At() / returns 2", func(t *testing.T) {
			ten, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := ten.At(); err != nil {
				t.Fatal(err)
			} else if int(val) != 2 {
				t.Fatalf("expected (2) as scalar tensor value, got (%f)", val)
			}
		})

		t.Run("Of([3]) 1D tensor / At(0) / returns 3", func(t *testing.T) {
			ten, err := tensor.Of([]float64{3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := ten.At(0); err != nil {
				t.Fatal(err)
			} else if int(val) != 3 {
				t.Fatalf("expected (3) as tensor value in position [0], got (%f)", val)
			}
		})

		t.Run("Of([1, 4]) 1D tensor / At(i) for all positions / returns 1 and 4", func(t *testing.T) {
			ten, err := tensor.Of([]float64{1., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := ten.At(0); err != nil {
				t.Fatal(err)
			} else if int(val) != 1 {
				t.Fatalf("expected (1) as tensor value in position [0], got (%f)", val)
			}

			if val, err := ten.At(1); err != nil {
				t.Fatal(err)
			} else if int(val) != 4 {
				t.Fatalf("expected (4) as tensor value in position [1], got (%f)", val)
			}
		})

		t.Run("Of([[-1], [-2]]) 2D tensor / At(i,j) for all positions / returns -1 and -2", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{{-1.}, {-2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := ten.At(0, 0); err != nil {
				t.Fatal(err)
			} else if int(val) != -1 {
				t.Fatalf("expected (-1) as tensor value in position [0,0], got (%f)", val)
			}

			if val, err := ten.At(1, 0); err != nil {
				t.Fatal(err)
			} else if int(val) != -2 {
				t.Fatalf("expected (-2) as tensor value in position [1,0], got (%f)", val)
			}
		})

		t.Run("Of(3x3x3 tensor) / At(i,j,k) for all positions / returns expected values", func(t *testing.T) {
			exp := [][][]float64{
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
			}

			ten, err := tensor.Of(exp, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			for i := range exp {
				for j := range exp[0] {
					for k := range exp[0][0] {
						if val, err := ten.At(i, j, k); err != nil {
							t.Fatal(err)
						} else if int(val) != int(exp[i][j][k]) {
							t.Fatalf("expected (%f) as tensor value in position [%d,%d,%d], got (%f)",
								exp[i][j][k], i, j, k, val)
						}
					}
				}
			}
		})

		t.Run("Of(1x2x3x4 tensor) / At(i,j,k,u) for all positions / returns 1 through 4 per row", func(t *testing.T) {
			exp := [][][][]float64{
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
			}

			ten, err := tensor.Of(exp, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			for i := range exp {
				for j := range exp[0] {
					for k := range exp[0][0] {
						for u := range exp[0][0][0] {
							if val, err := ten.At(i, j, k, u); err != nil {
								t.Fatal(err)
							} else if int(val) != int(exp[i][j][k][u]) {
								t.Fatalf("expected (%f) as tensor value in position [%d,%d,%d,%d], got (%f)",
									exp, i, j, k, u, val)
							}
						}
					}
				}
			}
		})

		// ============================== side effects ==============================

		t.Run("Of([5]) does not share 1D input slice / At(0) after mutating source / returns 5", func(t *testing.T) {
			d1 := []float64{5.}

			ten, err := tensor.Of(d1, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			d1[0] = 3.

			if val, err := ten.At(0); err != nil {
				t.Fatal(err)
			} else if int(val) != 5 {
				t.Fatalf("expected (5) as tensor value in position [0], got (%f)", val)
			}
		})

		t.Run("Of([[[[5]]]]) does not share 4D input slice / At(0,0,0,0) after mutating source / returns 5", func(t *testing.T) {
			d4 := [][][][]float64{{{{5.}}}}

			ten, err := tensor.Of(d4, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			d4[0][0][0][0] = 3.

			if val, err := ten.At(0, 0, 0, 0); err != nil {
				t.Fatal(err)
			} else if int(val) != 5 {
				t.Fatalf("expected (5) as tensor value in position [0,0,0,0], got (%f)", val)
			}
		})

		// ============================== validations ==============================

		t.Run("Of([]float64{}) / returns error: zero length along dimension", func(t *testing.T) {
			_, err := tensor.Of([]float64{}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of zero len along dimension (0)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: expected data to not have zero length along any dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][]float64{}) / returns error: zero length along dimension", func(t *testing.T) {
			_, err := tensor.Of([][]float64{}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of zero len along dimension (0)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: expected data to not have zero length along any dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][][]float64{}) / returns error: zero length along dimension", func(t *testing.T) {
			_, err := tensor.Of([][][]float64{}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of zero len along dimension (0)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: expected data to not have zero length along any dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][][][]float64{}) / returns error: zero length along dimension", func(t *testing.T) {
			_, err := tensor.Of([][][][]float64{}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of zero len along dimension (0)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: expected data to not have zero length along any dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][]float64{{}, {}}) / returns error: zero length along inner dimension", func(t *testing.T) {
			_, err := tensor.Of([][]float64{{}, {}}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of zero len along dimension (1)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: ValidateInputDataDimUnity input data validation failed: expected data to not have zero length along any dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][]float64{{}, {-1}}) / returns error: zero length along inner dimension", func(t *testing.T) {
			_, err := tensor.Of([][]float64{{}, {-1.}}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of zero len along dimension (1)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: ValidateInputDataDimUnity input data validation failed: expected data to not have zero length along any dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][][]float64{{{}}}) / returns error: zero length along inner dimension", func(t *testing.T) {
			_, err := tensor.Of([][][]float64{{{}}}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of zero len along dimension (1)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: ValidateInputDataDimUnity input data validation failed: ValidateInputDataDimUnity input data validation failed: expected data to not have zero length along any dimension", dev) {
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
				t.Fatal("expected error because of inconsistent tensor len along dimension (2)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: ValidateInputDataDimUnity input data validation failed: expected data to have have equal length along every dimension", dev) {
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
				t.Fatal("expected error because of inconsistent tensor len along dimension (2)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: ValidateInputDataDimUnity input data validation failed: expected data to have have equal length along every dimension", dev) {
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
				t.Fatal("expected error because of inconsistent tensor len along dimension (2)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: expected data to have have equal length along every dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([1]) with invalid device / returns error: invalid device", func(t *testing.T) {
			_, err := tensor.Of([]float64{1}, &tensor.Config{Device: -1})
			if err == nil {
				t.Fatal("expected error because of invalid input device")
			} else if err.Error() != "Of tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestEquals(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 3) scalar == Full(nil, 3) scalar / Equals() / returns true", func(t *testing.T) {
			t1, err := tensor.Full(nil, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			t2, err := tensor.Full(nil, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := t1.Equals(t2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected equal scalar tensors to be equal")
			}
		})

		t.Run("Full(nil, 3) scalar != Full(nil, 4) scalar / Equals() / returns false", func(t *testing.T) {
			t1, err := tensor.Full(nil, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			t2, err := tensor.Full(nil, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := t1.Equals(t2); err != nil {
				t.Fatal(err)
			} else if eq {
				t.Fatal("expected scalar tensors with different values to not be equal")
			}
		})

		t.Run("Of([1,2,3]) == Of([1,2,3]) 1D tensors / Equals() / returns true", func(t *testing.T) {
			t1, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			t2, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := t1.Equals(t2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected equal 1D tensors to be equal")
			}
		})

		t.Run("Of([1,2,3]) != Of([1,2,4]) 1D tensors / Equals() / returns false", func(t *testing.T) {
			t1, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			t2, err := tensor.Of([]float64{1., 2., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := t1.Equals(t2); err != nil {
				t.Fatal(err)
			} else if eq {
				t.Fatal("expected 1D tensors with a differing element to not be equal")
			}
		})

		t.Run("Of(2x3 matrix) == Of(2x3 same matrix) / Equals() / returns true", func(t *testing.T) {
			t1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			t2, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := t1.Equals(t2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected equal 2D tensors to be equal")
			}
		})

		t.Run("Of(2x3 matrix) != Of(2x3 different matrix) / Equals() / returns false", func(t *testing.T) {
			t1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			t2, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := t1.Equals(t2); err != nil {
				t.Fatal(err)
			} else if eq {
				t.Fatal("expected 2D tensors with a differing element to not be equal")
			}
		})

		t.Run("Full([2,3,4,2], 7) 4D tensor / Equals(itself) / returns true", func(t *testing.T) {
			ten, err := tensor.Full([]int{2, 3, 4, 2}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := ten.Equals(ten); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensor to equal itself")
			}
		})

		// ============================== validations ==============================

		t.Run("Full(nil) scalar / Equals(nil) / returns error: nil input tensor", func(t *testing.T) {
			t1, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = t1.Equals(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Equals tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([6,5,2]) / Equals(Full([6,4,2])) / returns error: size mismatch at dimension 1", func(t *testing.T) {
			t1, err := tensor.Full([]int{6, 5, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			t2, err := tensor.Full([]int{6, 4, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = t1.Equals(t2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Equals tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,5] / Equals / returns error: number of dimensions mismatch", func(t *testing.T) {
			t1, err := tensor.Zeros([]int{6, 5, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Zeros([]int{6, 5}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = t1.Equals(t2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Equals tensors' dimension validation failed: expected number of dimensions to match: (3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
