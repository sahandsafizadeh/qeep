package forward_test

import (
	"slices"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestZeros(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Zeros(nil) scalar / Equals Full(nil, 0.) / returns true", func(t *testing.T) {
			act, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Zeros([1]) 1-element 1D tensor / Equals Full([1], 0.) / returns true", func(t *testing.T) {
			act, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Zeros([3,4]) 2D tensor / Equals Full([3,4], 0.) / returns true", func(t *testing.T) {
			act, err := tensor.Zeros([]int{3, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Zeros([2,3,4]) 3D tensor / Equals Full([2,3,4], 0.) / returns true", func(t *testing.T) {
			act, err := tensor.Zeros([]int{2, 3, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{2, 3, 4}, 0., &tensor.Config{Device: dev})
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

		t.Run("Zeros([3,4]) does not share dims slice / Shape() after mutating dims / returns [3,4]", func(t *testing.T) {
			dims := []int{3, 4}

			ten, err := tensor.Zeros(dims, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			dims[0] = 1
			dims[1] = 1

			if shape := ten.Shape(); !slices.Equal(shape, []int{3, 4}) {
				t.Fatalf("expected tensor to have shape [3, 4], got %v", shape)
			}
		})

		// ============================== validations ==============================

		t.Run("Zeros([-1]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Zeros([]int{-1}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of non-positive dimension")
			} else if err.Error() != "Zeros input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([0]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Zeros([]int{0}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of non-positive dimension")
			} else if err.Error() != "Zeros input dimension validation failed: expected positive dimension sizes: got (0) at position (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1,-2]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Zeros([]int{1, -2}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of non-positive dimension")
			} else if err.Error() != "Zeros input dimension validation failed: expected positive dimension sizes: got (-2) at position (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([2,0,1]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Zeros([]int{2, 0, 1}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of non-positive dimension")
			} else if err.Error() != "Zeros input dimension validation failed: expected positive dimension sizes: got (0) at position (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1,1,1,1,1,1,1]) / returns error: too many dimensions", func(t *testing.T) {
			_, err := tensor.Zeros([]int{1, 1, 1, 1, 1, 1, 1}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of too many dimensions")
			} else if err.Error() != "Zeros input dimension validation failed: expected at most (6) dimensions: got (7)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros(nil) with invalid device / returns error: invalid device", func(t *testing.T) {
			_, err := tensor.Zeros(nil, &tensor.Config{Device: -1})
			if err == nil {
				t.Fatalf("expected error because of invalid input device")
			} else if err.Error() != "Zeros tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestOnes(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Ones(nil) scalar / Equals Full(nil, 1.) / returns true", func(t *testing.T) {
			act, err := tensor.Ones(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Ones([1]) 1-element 1D tensor / Equals Full([1], 1.) / returns true", func(t *testing.T) {
			act, err := tensor.Ones([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Ones([3,4]) 2D tensor / Equals Full([3,4], 1.) / returns true", func(t *testing.T) {
			act, err := tensor.Ones([]int{3, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Ones([2,3,4]) 3D tensor / Equals Full([2,3,4], 1.) / returns true", func(t *testing.T) {
			act, err := tensor.Ones([]int{2, 3, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{2, 3, 4}, 1., &tensor.Config{Device: dev})
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

		t.Run("Ones([3,4]) does not share dims slice / Shape() after mutating dims / returns [3,4]", func(t *testing.T) {
			dims := []int{3, 4}

			ten, err := tensor.Ones(dims, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			dims[0] = 1
			dims[1] = 1

			if shape := ten.Shape(); !slices.Equal(shape, []int{3, 4}) {
				t.Fatalf("expected tensor to have shape [3, 4], got %v", shape)
			}
		})

		// ============================== validations ==============================

		t.Run("Ones([-1]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Ones([]int{-1}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of non-positive dimension")
			} else if err.Error() != "Ones input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Ones([0]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Ones([]int{0}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of non-positive dimension")
			} else if err.Error() != "Ones input dimension validation failed: expected positive dimension sizes: got (0) at position (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Ones([1,-2]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Ones([]int{1, -2}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of non-positive dimension")
			} else if err.Error() != "Ones input dimension validation failed: expected positive dimension sizes: got (-2) at position (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Ones([2,0,1]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Ones([]int{2, 0, 1}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of non-positive dimension")
			} else if err.Error() != "Ones input dimension validation failed: expected positive dimension sizes: got (0) at position (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Ones([1,1,1,1,1,1,1]) / returns error: too many dimensions", func(t *testing.T) {
			_, err := tensor.Ones([]int{1, 1, 1, 1, 1, 1, 1}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatalf("expected error because of too many dimensions")
			} else if err.Error() != "Ones input dimension validation failed: expected at most (6) dimensions: got (7)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Ones(nil) with invalid device / returns error: invalid device", func(t *testing.T) {
			_, err := tensor.Ones(nil, &tensor.Config{Device: -1})
			if err == nil {
				t.Fatalf("expected error because of invalid input device")
			} else if err.Error() != "Ones tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

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

func TestConcat(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Zeros([3]) and Zeros([5]) / Concat(dim=0) / returns Zeros([8])", func(t *testing.T) {
			t1, err := tensor.Zeros([]int{3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Zeros([]int{5}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := tensor.Concat([]tensor.Tensor{t1, t2}, 0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{8}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Zeros([1,5,3]), Zeros([3,5,3]), Zeros([2,5,3]), Zeros([4,5,3]) / Concat(dim=0) / returns Zeros([10,5,3])", func(t *testing.T) {
			t1, err := tensor.Zeros([]int{1, 5, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Zeros([]int{3, 5, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Zeros([]int{2, 5, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t4, err := tensor.Zeros([]int{4, 5, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := tensor.Concat([]tensor.Tensor{t1, t2, t3, t4}, 0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{10, 5, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Zeros([4,2,3]), Zeros([4,4,3]), Zeros([4,1,3]), Zeros([4,3,3]) / Concat(dim=1) / returns Zeros([4,10,3])", func(t *testing.T) {
			t1, err := tensor.Zeros([]int{4, 2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Zeros([]int{4, 4, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Zeros([]int{4, 1, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t4, err := tensor.Zeros([]int{4, 3, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := tensor.Concat([]tensor.Tensor{t1, t2, t3, t4}, 1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{4, 10, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Three copies of 1x2x3 tensor / Concat(dim=0) / returns 3x2x3 tensor", func(t *testing.T) {
			t1, err := tensor.Of([][][]float64{
				{
					{0., 1., 2.},
					{3., 4., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := t1.Slice(nil)
			if err != nil {
				t.Fatal(err)
			}
			t3, err := t1.Slice(nil)
			if err != nil {
				t.Fatal(err)
			}

			act, err := tensor.Concat([]tensor.Tensor{t1, t2, t3}, 0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
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

		t.Run("Three copies of 1x2x3 tensor / Concat(dim=1) / returns 1x6x3 tensor", func(t *testing.T) {
			t1, err := tensor.Of([][][]float64{
				{
					{0., 1., 2.},
					{3., 4., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := t1.Slice(nil)
			if err != nil {
				t.Fatal(err)
			}
			t3, err := t1.Slice(nil)
			if err != nil {
				t.Fatal(err)
			}

			act, err := tensor.Concat([]tensor.Tensor{t1, t2, t3}, 1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{0., 1., 2.},
					{3., 4., 5.},
					{0., 1., 2.},
					{3., 4., 5.},
					{0., 1., 2.},
					{3., 4., 5.},
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

		t.Run("Three copies of 1x2x3 tensor / Concat(dim=2) / returns 1x2x9 tensor", func(t *testing.T) {
			t1, err := tensor.Of([][][]float64{
				{
					{0., 1., 2.},
					{3., 4., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := t1.Slice(nil)
			if err != nil {
				t.Fatal(err)
			}
			t3, err := t1.Slice(nil)
			if err != nil {
				t.Fatal(err)
			}

			act, err := tensor.Concat([]tensor.Tensor{t1, t2, t3}, 2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 0., 1., 2., 0., 1., 2.},
					{3., 4., 5., 3., 4., 5., 3., 4., 5.},
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

		t.Run("Concat([Zeros([4]), Zeros([6])], 0) does not share input slice / Concat then mutating slice / returns Zeros([10])", func(t *testing.T) {
			t1, err := tensor.Zeros([]int{4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			t2, err := tensor.Zeros([]int{6}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			ts := []tensor.Tensor{t1, t2}

			act, err := tensor.Concat(ts, 0)
			if err != nil {
				t.Fatal(err)
			}

			ts[1], err = tensor.Ones([]int{6}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{10}, &tensor.Config{Device: dev})
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

		t.Run("Concat(nil, 0) / returns error: fewer than 2 tensors", func(t *testing.T) {
			_, err := tensor.Concat(nil, 0)
			if err == nil {
				t.Fatalf("expected error because of the number of input tensors being less than (2)")
			} else if err.Error() != "Concat tensor implementation validation failed: expected at least (2) tensors: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([nil], 0) / returns error: fewer than 2 tensors", func(t *testing.T) {
			_, err := tensor.Concat([]tensor.Tensor{nil}, 0)
			if err == nil {
				t.Fatalf("expected error because of the number of input tensors being less than (2)")
			} else if err.Error() != "Concat tensor implementation validation failed: expected at least (2) tensors: got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([nil, nil], 0) / returns error: unsupported tensor implementation", func(t *testing.T) {
			_, err := tensor.Concat([]tensor.Tensor{nil, nil}, 0)
			if err == nil {
				t.Fatalf("expected error because of nil input tensors")
			} else if err.Error() != "Concat tensor implementation validation failed: unsupported tensor implementation" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([scalar, scalar], 0) / returns error: scalar tensor cannot be concatenated", func(t *testing.T) {
			t1, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2}, 0)
			if err == nil {
				t.Fatalf("expected error because of having scalar tensors as input")
			} else if err.Error() != "Concat inputs' dimension validation failed: scalar tensor can not be concatenated: got tensor (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Zeros([2]), Zeros([2]), Zeros([2,2])], 0) / returns error: tensors have different number of dimensions", func(t *testing.T) {
			t1, err := tensor.Zeros([]int{2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Zeros([]int{2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Zeros([]int{2, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 0)
			if err == nil {
				t.Fatalf("expected error because of the input tensors not having equal number of dimensions")
			} else if err.Error() != "Concat inputs' dimension validation failed: expected tensors to have the same number of dimensions: (2) != (1) for tensor (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Zeros([1]), Zeros([3])], -1) / returns error: dimension out of range [0,1)", func(t *testing.T) {
			t1, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Zeros([]int{3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2}, -1)
			if err == nil {
				t.Fatalf("expected error because of negative dimension")
			} else if err.Error() != "Concat inputs' dimension validation failed: expected concat dimension to be in range [0,1): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Zeros([1]), Zeros([3])], 1) / returns error: dimension out of range [0,1)", func(t *testing.T) {
			t1, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Zeros([]int{3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2}, 1)
			if err == nil {
				t.Fatalf("expected error because of dimension (1) being out of range")
			} else if err.Error() != "Concat inputs' dimension validation failed: expected concat dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Zeros([3,3]), Zeros([3,3])], 2) / returns error: dimension out of range [0,2)", func(t *testing.T) {
			t1, err := tensor.Zeros([]int{3, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Zeros([]int{3, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2}, 2)
			if err == nil {
				t.Fatalf("expected error because of dimension (2) being out of range")
			} else if err.Error() != "Concat inputs' dimension validation failed: expected concat dimension to be in range [0,2): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Zeros([2,2,2]), Zeros([2,2,1]), Zeros([3,2,2])], 0) / returns error: size mismatch at dim 2", func(t *testing.T) {
			t1, err := tensor.Zeros([]int{2, 2, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Zeros([]int{2, 2, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Zeros([]int{3, 2, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 0)
			if err == nil {
				t.Fatalf("expected error because of size mismatch along dimension (2)")
			} else if err.Error() != "Concat inputs' dimension validation failed: expected tensor sizes to match in all dimensions except (0): (1) != (2) for dimension (2) for tensor (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Zeros([2,1,2]), Zeros([2,2,2]), Zeros([3,2,2])], 0) / returns error: size mismatch at dim 1", func(t *testing.T) {
			t1, err := tensor.Zeros([]int{2, 1, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Zeros([]int{2, 2, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Zeros([]int{3, 2, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 0)
			if err == nil {
				t.Fatalf("expected error because of size mismatch along dimension (1)")
			} else if err.Error() != "Concat inputs' dimension validation failed: expected tensor sizes to match in all dimensions except (0): (2) != (1) for dimension (1) for tensor (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Zeros([2,1,2]), Zeros([1,2,2]), Zeros([2,3,2])], 1) / returns error: size mismatch at dim 0", func(t *testing.T) {
			t1, err := tensor.Zeros([]int{2, 1, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Zeros([]int{1, 2, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Zeros([]int{2, 3, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 1)
			if err == nil {
				t.Fatalf("expected error because of size mismatch along dimension (0)")
			} else if err.Error() != "Concat inputs' dimension validation failed: expected tensor sizes to match in all dimensions except (1): (1) != (2) for dimension (0) for tensor (1)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
