package forward_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestTranspose(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("1x1 matrix / Transpose() / returns same matrix", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{{1.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Transpose()
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
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1x3 row tensor / Transpose() / returns 3x1 column tensor", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{{1., 0., 2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Transpose()
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{{1.}, {0.}, {2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3x1 column tensor / Transpose() / returns 1x3 row tensor", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{{-2.}, {0.}, {-1.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Transpose()
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{{-2., 0., -1.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3x4 matrix / Transpose() / returns 4x3 transposed matrix", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{0., 1., 2., 3.},
				{0., 1., 2., 3.},
				{0., 1., 2., 3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Transpose()
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{0., 0., 0.},
				{1., 1., 1.},
				{2., 2., 2.},
				{3., 3., 3.},
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

		t.Run("3D tensor shape [2,2,2] / Transpose() / transposes last two dimensions", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2.},
					{3., 4.},
				},
				{
					{1., 2.},
					{3., 4.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Transpose()
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{1., 3.},
					{2., 4.},
				},
				{
					{1., 3.},
					{2., 4.},
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

		t.Run("4D tensor shape [2,2,2,3] / Transpose() / transposes last two dims leaving batch dims unchanged", func(t *testing.T) {
			ten, err := tensor.Of([][][][]float64{
				{
					{
						{1., 2., 3.},
						{4., 5., 6.},
					},
					{
						{7., 8., 9.},
						{10., 11., 12.},
					},
				},
				{
					{
						{13., 14., 15.},
						{16., 17., 18.},
					},
					{
						{19., 20., 21.},
						{22., 23., 24.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Transpose()
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][][]float64{
				{
					{
						{1., 4.},
						{2., 5.},
						{3., 6.},
					},
					{
						{7., 10.},
						{8., 11.},
						{9., 12.},
					},
				},
				{
					{
						{13., 16.},
						{14., 17.},
						{15., 18.},
					},
					{
						{19., 22.},
						{20., 23.},
						{21., 24.},
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

		// ============================== validations ==============================

		t.Run("Zeros(nil) scalar / Transpose() / returns error: fewer than 2 dimensions", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Transpose()
			if err == nil {
				t.Fatal("expected error because of tensor having less than 2 dimensions")
			} else if err.Error() != "Transpose tensor's dimension validation failed: expected tensor to have at least (2) dimensions for transpose: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([3]) 1D tensor / Transpose() / returns error: fewer than 2 dimensions", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Transpose()
			if err == nil {
				t.Fatal("expected error because of tensor having less than 2 dimensions")
			} else if err.Error() != "Transpose tensor's dimension validation failed: expected tensor to have at least (2) dimensions for transpose: got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestReshape(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Zeros(nil) scalar / Reshape([1,1]) / returns [1,1] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Reshape([]int{1, 1})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([1]) 1D tensor / Reshape(nil) / returns scalar tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Reshape(nil)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([1,1,1,1]) 4D tensor / Reshape([1,1]) / returns [1,1] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1, 1, 1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Reshape([]int{1, 1})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([4]) 1D tensor / Reshape([1,4]) / returns [1,4] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Reshape([]int{1, 4})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{1, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([4]) 1D tensor / Reshape([4,1]) / returns [4,1] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Reshape([]int{4, 1})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{4, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([4]) 1D tensor / Reshape([2,2]) / returns [2,2] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Reshape([]int{2, 2})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{2, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([1,2,3]) 3D tensor / Reshape([6]) / returns 1D tensor with 6 elements", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1, 2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Reshape([]int{6})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{6}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([1,2,3]) 3D tensor / Reshape([1,6,1]) / returns [1,6,1] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1, 2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Reshape([]int{1, 6, 1})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{1, 6, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([1,2,3]) 3D tensor / Reshape([3,2]) / returns [3,2] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1, 2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Reshape([]int{3, 2})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{3, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3,3]) 3D tensor with sequential elements 0..17 / Reshape([3,6]) / elements laid out row-major across new shape", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{0., 1., 2.},
					{3., 4., 5.},
					{6., 7., 8.},
				},
				{
					{9., 10., 11.},
					{12., 13., 14.},
					{15., 16., 17.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Reshape([]int{3, 6})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{0., 1., 2., 3., 4., 5.},
				{6., 7., 8., 9., 10., 11.},
				{12., 13., 14., 15., 16., 17.},
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

		// ============================== side effects ==============================

		t.Run("Zeros([2,3]) does not share shape slice / Reshape([3,2]) after mutating shape / returns correct [3,2] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			shape := []int{3, 2}

			act, err := ten.Reshape(shape)
			if err != nil {
				t.Fatal(err)
			}

			shape[0] = 1
			shape[1] = 6

			exp, err := tensor.Zeros([]int{3, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== validations ==============================

		t.Run("Zeros([3,2]) / Reshape([2,3,-1]) / returns error: non-positive dimension", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Reshape([]int{2, 3, -1})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != "Reshape input shape validation failed: expected positive dimension sizes: got (-1) at position (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([3,2]) / Reshape([1,1,1,1,1,1,1]) / returns error: too many dimensions", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Reshape([]int{1, 1, 1, 1, 1, 1, 1})
			if err == nil {
				t.Fatal("expected error because of too many dimensions")
			} else if err.Error() != "Reshape input shape validation failed: expected at most (6) dimensions: got (7)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros(nil) scalar / Reshape([2]) / returns error: incompatible number of elements", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Reshape([]int{2})
			if err == nil {
				t.Fatal("expected error because of incompatible number of elements in source (1) with target (2)")
			} else if err.Error() != "Reshape input shape validation failed: expected number of elements in source and target tensors to match: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([2]) 1D tensor / Reshape([2,3]) / returns error: incompatible number of elements", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Reshape([]int{2, 3})
			if err == nil {
				t.Fatal("expected error because of incompatible number of elements in source (2) with target (6)")
			} else if err.Error() != "Reshape input shape validation failed: expected number of elements in source and target tensors to match: (2) != (6)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestUnSqueeze(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Zeros(nil) scalar / UnSqueeze(0) / returns [1] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.UnSqueeze(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([2]) 1D tensor / UnSqueeze(0) / returns [1,2] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.UnSqueeze(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{1, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([2]) 1D tensor / UnSqueeze(1) / returns [2,1] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.UnSqueeze(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{2, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([2,3]) 2D tensor / UnSqueeze(0) / returns [1,2,3] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.UnSqueeze(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{1, 2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([2,3]) 2D tensor / UnSqueeze(1) / returns [2,1,3] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.UnSqueeze(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{2, 1, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([2,3]) 2D tensor / UnSqueeze(2) / returns [2,3,1] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.UnSqueeze(2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{2, 3, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3]) 2D tensor with distinct values / UnSqueeze(1) / returns [2,1,3] tensor with values preserved", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.UnSqueeze(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{{1., 2., 3.}},
				{{4., 5., 6.}},
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

		// ============================== side effects ==============================

		t.Run("Of([2,3]) 2D tensor with values 1..6 / UnSqueeze(0) twice / original tensor unchanged; result adds extra leading dimensions", func(t *testing.T) {
			t1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := t1.UnSqueeze(0)
			if err != nil {
				t.Fatal(err)
			}

			exp1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err := tensor.Of([][][]float64{
				{
					{1., 2., 3.},
					{4., 5., 6.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := t1.Equals(exp1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := t2.Equals(exp2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			t2, err = t2.UnSqueeze(0)
			if err != nil {
				t.Fatal(err)
			}

			exp1, err = tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err = tensor.Of([][][][]float64{
				{
					{
						{1., 2., 3.},
						{4., 5., 6.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := t1.Equals(exp1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := t2.Equals(exp2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== validations ==============================

		t.Run("Zeros(nil) scalar / UnSqueeze(-1) / returns error: dimension out of range [0,0]", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.UnSqueeze(-1)
			if err == nil {
				t.Fatal("expected error because of dimension (-1) being out of range")
			} else if err.Error() != "UnSqueeze input dimension validation failed: expected dimension to be in range [0,0]: got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros(nil) scalar / UnSqueeze(1) / returns error: dimension out of range [0,0]", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.UnSqueeze(1)
			if err == nil {
				t.Fatal("expected error because of dimension (1) being out of range")
			} else if err.Error() != "UnSqueeze input dimension validation failed: expected dimension to be in range [0,0]: got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([2]) 1D tensor / UnSqueeze(2) / returns error: dimension out of range [0,1]", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.UnSqueeze(2)
			if err == nil {
				t.Fatal("expected error because of dimension (2) being out of range")
			} else if err.Error() != "UnSqueeze input dimension validation failed: expected dimension to be in range [0,1]: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1,1,1,1,1,1]) 6D tensor / UnSqueeze(2) / returns error: exceeds maximum 6 dimensions", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1, 1, 1, 1, 1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.UnSqueeze(2)
			if err == nil {
				t.Fatal("expected error because of too many dimensions")
			} else if err.Error() != "UnSqueeze input dimension validation failed: operation causes tensor to exceed maximum (6) dimensions" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestSqueeze(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Zeros([1]) 1D tensor / Squeeze(0) / returns scalar tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Squeeze(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([1,2,3]) 3D tensor / Squeeze(0) / returns [2,3] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1, 2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Squeeze(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([2,1,3]) 3D tensor / Squeeze(1) / returns [2,3] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2, 1, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Squeeze(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([2,3,1]) 3D tensor / Squeeze(2) / returns [2,3] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2, 3, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Squeeze(2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,1,3]) 3D tensor with distinct values / Squeeze(1) / returns [2,3] tensor with values preserved", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{{1., 2., 3.}},
				{{4., 5., 6.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Squeeze(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
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

		// ============================== side effects ==============================

		t.Run("Of([1,1,3]) 3D tensor with values 1..3 / Squeeze(0) twice / original tensor unchanged; result reduces to 1D tensor", func(t *testing.T) {
			t1, err := tensor.Of([][][]float64{
				{{1., 2., 3.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := t1.Squeeze(0)
			if err != nil {
				t.Fatal(err)
			}

			exp1, err := tensor.Of([][][]float64{
				{{1., 2., 3.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err := tensor.Of([][]float64{
				{1., 2., 3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := t1.Equals(exp1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := t2.Equals(exp2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			t2, err = t2.Squeeze(0)
			if err != nil {
				t.Fatal(err)
			}

			exp1, err = tensor.Of([][][]float64{
				{{1., 2., 3.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err = tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := t1.Equals(exp1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := t2.Equals(exp2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== validations ==============================

		t.Run("Zeros(nil) scalar / Squeeze(-1) / returns error: dimension out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Squeeze(-1)
			if err == nil {
				t.Fatal("expected error because of dimension (-1) being out of range")
			} else if err.Error() != "Squeeze input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros(nil) scalar / Squeeze(0) / returns error: dimension out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Squeeze(0)
			if err == nil {
				t.Fatal("expected error because of dimension (0) being out of range")
			} else if err.Error() != "Squeeze input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1]) 1D tensor / Squeeze(1) / returns error: dimension out of range [0,1)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Squeeze(1)
			if err == nil {
				t.Fatal("expected error because of dimension (1) being out of range")
			} else if err.Error() != "Squeeze input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1,2,3]) 3D tensor / Squeeze(3) / returns error: dimension out of range [0,3)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1, 2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Squeeze(3)
			if err == nil {
				t.Fatal("expected error because of dimension (3) being out of range")
			} else if err.Error() != "Squeeze input dimension validation failed: expected dimension to be in range [0,3): got (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1,2,3]) 3D tensor / Squeeze(2) / returns error: squeeze dimension size is 3, not 1", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1, 2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Squeeze(2)
			if err == nil {
				t.Fatal("expected error because of dimension (2) not being equal to (1)")
			} else if err.Error() != "Squeeze input dimension validation failed: expected squeeze dimension to be (1): got (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1,2,3]) 3D tensor / Squeeze(1) / returns error: squeeze dimension size is 2, not 1", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1, 2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Squeeze(1)
			if err == nil {
				t.Fatal("expected error because of dimension (1) not being equal to (1)")
			} else if err.Error() != "Squeeze input dimension validation failed: expected squeeze dimension to be (1): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestFlatten(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Zeros([1]) 1D tensor / Flatten(0) / returns [1] 1D tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Flatten(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([2,3,4]) 3D tensor / Flatten(0) / returns [24] 1D tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2, 3, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Flatten(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{24}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([2,3,4]) 3D tensor / Flatten(1) / returns [2,12] 2D tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2, 3, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Flatten(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{2, 12}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([2,3,4]) 3D tensor / Flatten(2) / returns same shape [2,3,4]", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2, 3, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Flatten(2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{2, 3, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([1,2,3,4,5,6]) 6D tensor / Flatten(2) / returns [1,2,360] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1, 2, 3, 4, 5, 6}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Flatten(2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Zeros([]int{1, 2, 360}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3,4]) 3D tensor with sequential values 0..23 / Flatten(1) / returns [2,12] tensor with values preserved in row-major order", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Flatten(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.},
				{12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.},
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

		// ============================== side effects ==============================

		t.Run("Of([2,3,4]) 3D tensor with sequential values 0..23 / Flatten(1) then Flatten(0) / original tensor unchanged; result is 1D tensor", func(t *testing.T) {
			t1, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			t2, err := t1.Flatten(1)
			if err != nil {
				t.Fatal(err)
			}

			exp1, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err := tensor.Of([][]float64{
				{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.},
				{12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := t1.Equals(exp1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := t2.Equals(exp2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			t2, err = t2.Flatten(0)
			if err != nil {
				t.Fatal(err)
			}

			exp1, err = tensor.Of([][][]float64{
				{
					{0., 1., 2., 3.},
					{4., 5., 6., 7.},
					{8., 9., 10., 11.},
				},
				{
					{12., 13., 14., 15.},
					{16., 17., 18., 19.},
					{20., 21., 22., 23.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err = tensor.Of([]float64{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := t1.Equals(exp1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := t2.Equals(exp2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== validations ==============================

		t.Run("Zeros(nil) scalar / Flatten(-1) / returns error: dimension out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Flatten(-1)
			if err == nil {
				t.Fatal("expected error because of dimension (-1) being out of range")
			} else if err.Error() != "Flatten input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros(nil) scalar / Flatten(0) / returns error: dimension out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Flatten(0)
			if err == nil {
				t.Fatal("expected error because of dimension (0) being out of range")
			} else if err.Error() != "Flatten input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1]) 1D tensor / Flatten(1) / returns error: dimension out of range [0,1)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Flatten(1)
			if err == nil {
				t.Fatal("expected error because of dimension (1) being out of range")
			} else if err.Error() != "Flatten input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1,2,3]) 3D tensor / Flatten(3) / returns error: dimension out of range [0,3)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1, 2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Flatten(3)
			if err == nil {
				t.Fatal("expected error because of dimension (3) being out of range")
			} else if err.Error() != "Flatten input dimension validation failed: expected dimension to be in range [0,3): got (3)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestBroadcast(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Of(5) scalar / Broadcast(nil) / returns same scalar", func(t *testing.T) {
			ten, err := tensor.Of(5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Broadcast(nil)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of(5) scalar / Broadcast([2,1]) / returns [2,1] tensor filled with 5", func(t *testing.T) {
			ten, err := tensor.Of(5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Broadcast([]int{2, 1})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{{5.}, {5.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([5]) 1D single-element tensor / Broadcast([3,2]) / returns [3,2] tensor filled with 5", func(t *testing.T) {
			ten, err := tensor.Of([]float64{5.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Broadcast([]int{3, 2})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{5., 5.},
				{5., 5.},
				{5., 5.},
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

		t.Run("Of([1,2]) 1D tensor / Broadcast([3,3,2]) / repeats values across new batch dimensions", func(t *testing.T) {
			ten, err := tensor.Of([]float64{1., 2.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Broadcast([]int{3, 3, 2})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
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

		t.Run("Of shape [2,1,1] with values 0 and 1 / Broadcast([2,3,4]) / each value broadcast over its [3,4] slice", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{{{0.}}, {{1.}}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Broadcast([]int{2, 3, 4})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
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

		t.Run("Of shape [2,1,4] with rows [0..3] and [4..7] / Broadcast([1,2,3,4]) / each row broadcast across 3 repetitions", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{{0., 1., 2., 3.}},
				{{4., 5., 6., 7.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Broadcast([]int{1, 2, 3, 4})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][][]float64{
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

		t.Run("Of shape [2,1,4] with rows [0..3] and [4..7] / Broadcast([2,1,4]) / returns same tensor unchanged", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{{0., 1., 2., 3.}},
				{{4., 5., 6., 7.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Broadcast([]int{2, 1, 4})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{{0., 1., 2., 3.}},
				{{4., 5., 6., 7.}},
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

		t.Run("Ones([4,1,1,3]) / Broadcast([6,5,4,3,3,3]) / returns all-ones [6,5,4,3,3,3] tensor", func(t *testing.T) {
			ten, err := tensor.Ones([]int{4, 1, 1, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Broadcast([]int{6, 5, 4, 3, 3, 3})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Ones([]int{6, 5, 4, 3, 3, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== side effects ==============================

		t.Run("Of([2,3]) 2D tensor with values 1..6 / Broadcast([2,3]) twice / original tensor unchanged; result equals original each time", func(t *testing.T) {
			t1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := t1.Broadcast([]int{2, 3})
			if err != nil {
				t.Fatal(err)
			}

			exp1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := t1.Equals(exp1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := t2.Equals(exp2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			t2, err = t2.Broadcast([]int{2, 3})
			if err != nil {
				t.Fatal(err)
			}

			exp1, err = tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err = tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := t1.Equals(exp1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := t2.Equals(exp2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([2,1]) does not share shape slice / Broadcast([2,3]) after mutating shape / returns correct [2,3] tensor", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			shape := []int{2, 3}

			act, err := ten.Broadcast(shape)
			if err != nil {
				t.Fatal(err)
			}

			shape[0] = 1
			shape[1] = 6

			exp, err := tensor.Zeros([]int{2, 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== validations ==============================

		t.Run("Zeros([3,2]) / Broadcast([3,-2]) / returns error: non-positive dimension", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Broadcast([]int{3, -2})
			if err == nil {
				t.Fatal("expected error because of negative dimension")
			} else if err.Error() != "Broadcast input shape validation failed: expected positive dimension sizes: got (-2) at position (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([3,2]) / Broadcast([1,1,1,1,1,1,1,1]) / returns error: too many dimensions", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Broadcast([]int{1, 1, 1, 1, 1, 1, 1, 1})
			if err == nil {
				t.Fatal("expected error because of too many dimensions")
			} else if err.Error() != "Broadcast input shape validation failed: expected at most (6) dimensions: got (8)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1,2]) 2D tensor / Broadcast(nil) / returns error: source has more dimensions than target", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Broadcast(nil)
			if err == nil {
				t.Fatal("expected error because of source number of dimensions (2) being greater than that of target (0)")
			} else if err.Error() != "Broadcast input shape validation failed: expected number of dimensions in source tensor to be less than or equal to that of target shape: (2) > (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1,2]) 2D tensor / Broadcast([2]) / returns error: source has more dimensions than target", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1, 2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Broadcast([]int{2})
			if err == nil {
				t.Fatal("expected error because of source number of dimensions (2) being greater than that of target (1)")
			} else if err.Error() != "Broadcast input shape validation failed: expected number of dimensions in source tensor to be less than or equal to that of target shape: (2) > (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([2]) 1D tensor / Broadcast([1]) / returns error: incompatible size at dimension 0", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Broadcast([]int{1})
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([2]) 1D tensor / Broadcast([3]) / returns error: incompatible size at dimension 0", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Broadcast([]int{3})
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([4,2,1]) / Broadcast([4,3,5]) / returns error: incompatible size at dimension 1", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{4, 2, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Broadcast([]int{4, 3, 5})
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([2,1,4,1]) / Broadcast([2,3,4,4,5]) / returns error: incompatible size at dimension 1", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{2, 1, 4, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Broadcast([]int{2, 3, 4, 4, 5})
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([3,1,3,6]) / Broadcast([1,2,3,4,5,6]) / returns error: incompatible size at dimension 4", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3, 1, 3, 6}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Broadcast([]int{1, 2, 3, 4, 5, 6})
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (4)")
			} else if err.Error() != "Broadcast input shape validation failed: expected target shape to be (3) or source size to be (1) at dimension (4): got shape (5)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
