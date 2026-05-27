package forward_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestSum(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("scalar tensor / Sum() / returns 9", func(t *testing.T) {
			ten, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Sum(); int(val) != 9 {
				t.Fatalf("expected (9) as the sum value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [6, 4] / Sum() / returns 10", func(t *testing.T) {
			ten, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Sum(); int(val) != 10 {
				t.Fatalf("expected (10) as the sum value of tensor, got (%f)", val)
			}
		})

		t.Run("3D tensor shape [3,1,4] / Sum() / returns 36", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{{9., -1., 8., 6.}},
				{{-5., 4., 1., 0.}},
				{{2., 8., 7., -3.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Sum(); int(val) != 36 {
				t.Fatalf("expected (36) as the sum value of tensor, got (%f)", val)
			}
		})
	})
}

func TestMax(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("scalar tensor / Max() / returns 9", func(t *testing.T) {
			ten, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Max(); int(val) != 9 {
				t.Fatalf("expected (9) as the max value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [6, 4] / Max() / returns 6", func(t *testing.T) {
			ten, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Max(); int(val) != 6 {
				t.Fatalf("expected (6) as the max value of tensor, got (%f)", val)
			}
		})

		t.Run("3D tensor shape [3,1,4] / Max() / returns 9", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{{9., -1., 8., 6.}},
				{{-5., 4., 1., 0.}},
				{{2., 8., 7., -3.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Max(); int(val) != 9 {
				t.Fatalf("expected (9) as the max value of tensor, got (%f)", val)
			}
		})
	})
}

func TestMin(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("scalar tensor / Min() / returns 9", func(t *testing.T) {
			ten, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Min(); int(val) != 9 {
				t.Fatalf("expected (9) as the min value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [6, 4] / Min() / returns 4", func(t *testing.T) {
			ten, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Min(); int(val) != 4 {
				t.Fatalf("expected (4) as the min value of tensor, got (%f)", val)
			}
		})

		t.Run("3D tensor shape [3,1,4] / Min() / returns -5", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{{9., -1., 8., 6.}},
				{{-5., 4., 1., 0.}},
				{{2., 8., 7., -3.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Min(); int(val) != -5 {
				t.Fatalf("expected (-5) as the min value of tensor, got (%f)", val)
			}
		})
	})
}

func TestAvg(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("scalar tensor / Avg() / returns 9", func(t *testing.T) {
			ten, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Avg(); int(val) != 9 {
				t.Fatalf("expected (9) as the avg value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [6, 4] / Avg() / returns 5", func(t *testing.T) {
			ten, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Avg(); int(val) != 5 {
				t.Fatalf("expected (5) as the avg value of tensor, got (%f)", val)
			}
		})

		t.Run("3D tensor shape [3,1,4] / Avg() / returns 3", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{{9., -1., 8., 6.}},
				{{-5., 4., 1., 0.}},
				{{2., 8., 7., -3.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Avg(); int(val) != 3 {
				t.Fatalf("expected (3) as the avg value of tensor, got (%f)", val)
			}
		})
	})
}

func TestVar(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("scalar tensor / Var() / returns 0", func(t *testing.T) {
			ten, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Var(); int(val) != 0 {
				t.Fatalf("expected (0) as the var value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [-sqrt2, sqrt2] / Var() / returns 4", func(t *testing.T) {
			ten, err := tensor.Of([]float64{-math.Sqrt2, math.Sqrt2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Var(); int(val) != 4 {
				t.Fatalf("expected (4) as the var value of tensor, got (%f)", val)
			}
		})

		t.Run("3D tensor shape [1,3,1] / Var() / returns 4", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{{{-2.}, {0.}, {2.}}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Var(); int(val) != 4 {
				t.Fatalf("expected (4) as the var value of tensor, got (%f)", val)
			}
		})
	})
}

func TestStd(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("scalar tensor / Std() / returns 0", func(t *testing.T) {
			ten, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Std(); int(val) != 0 {
				t.Fatalf("expected (0) as the std value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [-sqrt2, sqrt2] / Std() / returns 2", func(t *testing.T) {
			ten, err := tensor.Of([]float64{-math.Sqrt2, math.Sqrt2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Std(); int(val) != 2 {
				t.Fatalf("expected (2) as the std value of tensor, got (%f)", val)
			}
		})

		t.Run("3D tensor shape [1,3,1] / Std() / returns 2", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{{{-2.}, {0.}, {2.}}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Std(); int(val) != 2 {
				t.Fatalf("expected (2) as the std value of tensor, got (%f)", val)
			}
		})
	})
}

func TestMean(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("scalar tensor / Mean() / returns 9", func(t *testing.T) {
			ten, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Mean(); int(val) != 9 {
				t.Fatalf("expected (9) as the mean value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [6, 4] / Mean() / returns 5", func(t *testing.T) {
			ten, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Mean(); int(val) != 5 {
				t.Fatalf("expected (5) as the mean value of tensor, got (%f)", val)
			}
		})

		t.Run("3D tensor shape [3,1,4] / Mean() / returns 3", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{{9., -1., 8., 6.}},
				{{-5., 4., 1., 0.}},
				{{2., 8., 7., -3.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Mean(); int(val) != 3 {
				t.Fatalf("expected (3) as the mean value of tensor, got (%f)", val)
			}
		})
	})
}

func TestSumAlong(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Ones([1]) tensor / SumAlong(0) / returns scalar 1", func(t *testing.T) {
			ten, err := tensor.Ones([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.SumAlong(0)
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
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Ones([3]) tensor / SumAlong(0) / returns scalar 3", func(t *testing.T) {
			ten, err := tensor.Ones([]int{3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Ones([3,4,5]) tensor / SumAlong(0) / returns Full([4,5], 3)", func(t *testing.T) {
			ten, err := tensor.Ones([]int{3, 4, 5}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{4, 5}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Ones([3,4,5]) tensor / SumAlong(1) / returns Full([3,5], 4)", func(t *testing.T) {
			ten, err := tensor.Ones([]int{3, 4, 5}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.SumAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{3, 5}, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Ones([3,4,5]) tensor / SumAlong(2) / returns Full([3,4], 5)", func(t *testing.T) {
			ten, err := tensor.Ones([]int{3, 4, 5}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.SumAlong(2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{3, 4}, 5., &tensor.Config{Device: dev})
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

		t.Run("scalar tensor / SumAlong(-1) / returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.SumAlong(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "SumAlong input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor / SumAlong(0) / returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.SumAlong(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "SumAlong input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1]) tensor / SumAlong(1) / returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.SumAlong(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "SumAlong input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([3,1]) tensor / SumAlong(2) / returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.SumAlong(2)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (2) being out of range")
			} else if err.Error() != "SumAlong input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestMaxAlong(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Ones([1]) tensor / MaxAlong(0) / returns scalar 1", func(t *testing.T) {
			ten, err := tensor.Ones([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MaxAlong(0)
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
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Ones([3]) tensor / MaxAlong(0) / returns scalar 1", func(t *testing.T) {
			ten, err := tensor.Ones([]int{3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MaxAlong(0)
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
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("tensor shape [2,3,3] / MaxAlong(0) / returns element-wise max along dim 0", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2., -5.},
					{0., -1., 3.},
					{7., -7., 7.},
				},
				{
					{8., -1., 0.},
					{5., 4., -3.},
					{1., -3., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MaxAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{8., 2., 0.},
				{5., 4., 3.},
				{7., -3., 7.},
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

		t.Run("tensor shape [2,3,3] / MaxAlong(1) / returns element-wise max along dim 1", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2., -5.},
					{0., -1., 3.},
					{7., -7., 7.},
				},
				{
					{8., -1., 0.},
					{5., 4., -3.},
					{1., -3., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MaxAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{7., 2., 7.},
				{8., 4., 5.},
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

		t.Run("tensor shape [2,3,3] / MaxAlong(2) / returns element-wise max along dim 2", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2., -5.},
					{0., -1., 3.},
					{7., -7., 7.},
				},
				{
					{8., -1., 0.},
					{5., 4., -3.},
					{1., -3., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MaxAlong(2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{2., 3., 7.},
				{8., 5., 5.},
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

		t.Run("scalar tensor / MaxAlong(-1) / returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.MaxAlong(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "MaxAlong input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor / MaxAlong(0) / returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.MaxAlong(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "MaxAlong input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1]) tensor / MaxAlong(1) / returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.MaxAlong(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "MaxAlong input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([3,1]) tensor / MaxAlong(2) / returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.MaxAlong(2)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (2) being out of range")
			} else if err.Error() != "MaxAlong input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestMinAlong(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Ones([1]) tensor / MinAlong(0) / returns scalar 1", func(t *testing.T) {
			ten, err := tensor.Ones([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MinAlong(0)
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
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Ones([3]) tensor / MinAlong(0) / returns scalar 1", func(t *testing.T) {
			ten, err := tensor.Ones([]int{3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MinAlong(0)
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
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("tensor shape [2,3,3] / MinAlong(0) / returns element-wise min along dim 0", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2., -5.},
					{0., -1., 3.},
					{7., -7., 7.},
				},
				{
					{8., -1., 0.},
					{5., 4., -3.},
					{1., -3., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MinAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{1., -1., -5.},
				{0., -1., -3.},
				{1., -7., 5.},
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

		t.Run("tensor shape [2,3,3] / MinAlong(1) / returns element-wise min along dim 1", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2., -5.},
					{0., -1., 3.},
					{7., -7., 7.},
				},
				{
					{8., -1., 0.},
					{5., 4., -3.},
					{1., -3., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MinAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{0., -7., -5.},
				{1., -3., -3.},
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

		t.Run("tensor shape [2,3,3] / MinAlong(2) / returns element-wise min along dim 2", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2., -5.},
					{0., -1., 3.},
					{7., -7., 7.},
				},
				{
					{8., -1., 0.},
					{5., 4., -3.},
					{1., -3., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MinAlong(2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{-5., -1., -7.},
				{-1., -3., -3.},
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

		t.Run("scalar tensor / MinAlong(-1) / returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.MinAlong(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "MinAlong input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor / MinAlong(0) / returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.MinAlong(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "MinAlong input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1]) tensor / MinAlong(1) / returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.MinAlong(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "MinAlong input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([3,1]) tensor / MinAlong(2) / returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.MinAlong(2)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (2) being out of range")
			} else if err.Error() != "MinAlong input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestAvgAlong(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Ones([1]) tensor / AvgAlong(0) / returns scalar 1", func(t *testing.T) {
			ten, err := tensor.Ones([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.AvgAlong(0)
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
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3], 2) tensor / AvgAlong(0) / returns scalar 2", func(t *testing.T) {
			ten, err := tensor.Full([]int{3}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.AvgAlong(0)
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

		t.Run("Full([3,4,5], 3) tensor / AvgAlong(0) / returns Full([4,5], 3)", func(t *testing.T) {
			ten, err := tensor.Full([]int{3, 4, 5}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.AvgAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{4, 5}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,4,5], 4) tensor / AvgAlong(1) / returns Full([3,5], 4)", func(t *testing.T) {
			ten, err := tensor.Full([]int{3, 4, 5}, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.AvgAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{3, 5}, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,4,5], 5) tensor / AvgAlong(2) / returns Full([3,4], 5)", func(t *testing.T) {
			ten, err := tensor.Full([]int{3, 4, 5}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.AvgAlong(2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{3, 4}, 5., &tensor.Config{Device: dev})
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

		t.Run("scalar tensor / AvgAlong(-1) / returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.AvgAlong(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "AvgAlong input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor / AvgAlong(0) / returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.AvgAlong(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "AvgAlong input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1]) tensor / AvgAlong(1) / returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.AvgAlong(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "AvgAlong input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([3,1]) tensor / AvgAlong(2) / returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.AvgAlong(2)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (2) being out of range")
			} else if err.Error() != "AvgAlong input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestVarAlong(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Ones([1]) tensor / VarAlong(0) / returns scalar 0", func(t *testing.T) {
			ten, err := tensor.Ones([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.VarAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3], 5) tensor / VarAlong(0) / returns scalar 0", func(t *testing.T) {
			ten, err := tensor.Full([]int{3}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.VarAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(0., &tensor.Config{Device: dev})
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

		t.Run("scalar tensor / VarAlong(-1) / returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.VarAlong(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "VarAlong input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor / VarAlong(0) / returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.VarAlong(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "VarAlong input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1]) tensor / VarAlong(1) / returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.VarAlong(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "VarAlong input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([3,1]) tensor / VarAlong(2) / returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.VarAlong(2)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (2) being out of range")
			} else if err.Error() != "VarAlong input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestStdAlong(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Ones([1]) tensor / StdAlong(0) / returns scalar 0", func(t *testing.T) {
			ten, err := tensor.Ones([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.StdAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3], 5) tensor / StdAlong(0) / returns scalar 0", func(t *testing.T) {
			ten, err := tensor.Full([]int{3}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.StdAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(0., &tensor.Config{Device: dev})
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

		t.Run("scalar tensor / StdAlong(-1) / returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.StdAlong(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "StdAlong input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor / StdAlong(0) / returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.StdAlong(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "StdAlong input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1]) tensor / StdAlong(1) / returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.StdAlong(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "StdAlong input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([3,1]) tensor / StdAlong(2) / returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.StdAlong(2)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (2) being out of range")
			} else if err.Error() != "StdAlong input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestMeanAlong(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Ones([1]) tensor / MeanAlong(0) / returns scalar 1", func(t *testing.T) {
			ten, err := tensor.Ones([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MeanAlong(0)
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
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3], 2) tensor / MeanAlong(0) / returns scalar 2", func(t *testing.T) {
			ten, err := tensor.Full([]int{3}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MeanAlong(0)
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

		t.Run("Full([3,4,5], 3) tensor / MeanAlong(0) / returns Full([4,5], 3)", func(t *testing.T) {
			ten, err := tensor.Full([]int{3, 4, 5}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MeanAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{4, 5}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,4,5], 4) tensor / MeanAlong(1) / returns Full([3,5], 4)", func(t *testing.T) {
			ten, err := tensor.Full([]int{3, 4, 5}, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MeanAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{3, 5}, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,4,5], 5) tensor / MeanAlong(2) / returns Full([3,4], 5)", func(t *testing.T) {
			ten, err := tensor.Full([]int{3, 4, 5}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MeanAlong(2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{3, 4}, 5., &tensor.Config{Device: dev})
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

		t.Run("scalar tensor / MeanAlong(-1) / returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.MeanAlong(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "MeanAlong input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor / MeanAlong(0) / returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.MeanAlong(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "MeanAlong input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1]) tensor / MeanAlong(1) / returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.MeanAlong(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "MeanAlong input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([3,1]) tensor / MeanAlong(2) / returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.MeanAlong(2)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (2) being out of range")
			} else if err.Error() != "MeanAlong input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestArgmax(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Ones([1]) tensor / Argmax(0) / returns scalar 0", func(t *testing.T) {
			ten, err := tensor.Ones([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Argmax(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Ones([3]) tensor / Argmax(0) / returns scalar 0", func(t *testing.T) {
			ten, err := tensor.Ones([]int{3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Argmax(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("tensor shape [2,3,3] / Argmax(0) / returns argmax indices along dim 0", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2., -5.},
					{0., -1., 3.},
					{7., -7., 7.},
				},
				{
					{8., -1., 0.},
					{5., 4., -3.},
					{1., -3., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Argmax(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{1., 0., 1.},
				{1., 1., 0.},
				{0., 1., 0.},
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

		t.Run("tensor shape [2,3,3] / Argmax(1) / returns argmax indices along dim 1", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2., -5.},
					{0., -1., 3.},
					{7., -7., 7.},
				},
				{
					{8., -1., 0.},
					{5., 4., -3.},
					{1., -3., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Argmax(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{2., 0., 2.},
				{0., 1., 2.},
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

		t.Run("tensor shape [2,3,3] / Argmax(2) / returns argmax indices along dim 2", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2., -5.},
					{0., -1., 3.},
					{7., -7., 7.},
				},
				{
					{8., -1., 0.},
					{5., 4., -3.},
					{1., -3., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Argmax(2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{1., 2., 0.},
				{0., 0., 2.},
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

		t.Run("scalar tensor / Argmax(-1) / returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Argmax(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "Argmax input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor / Argmax(0) / returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Argmax(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "Argmax input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1]) tensor / Argmax(1) / returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Argmax(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "Argmax input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([3,1]) tensor / Argmax(2) / returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Argmax(2)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (2) being out of range")
			} else if err.Error() != "Argmax input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestArgmin(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Ones([1]) tensor / Argmin(0) / returns scalar 0", func(t *testing.T) {
			ten, err := tensor.Ones([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Argmin(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Ones([3]) tensor / Argmin(0) / returns scalar 0", func(t *testing.T) {
			ten, err := tensor.Ones([]int{3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Argmin(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("tensor shape [2,3,3] / Argmin(0) / returns argmin indices along dim 0", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2., -5.},
					{0., -1., 3.},
					{7., -7., 7.},
				},
				{
					{8., -1., 0.},
					{5., 4., -3.},
					{1., -3., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Argmin(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{0., 1., 0.},
				{0., 0., 1.},
				{1., 0., 1.},
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

		t.Run("tensor shape [2,3,3] / Argmin(1) / returns argmin indices along dim 1", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2., -5.},
					{0., -1., 3.},
					{7., -7., 7.},
				},
				{
					{8., -1., 0.},
					{5., 4., -3.},
					{1., -3., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Argmin(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{1., 2., 0.},
				{2., 2., 1.},
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

		t.Run("tensor shape [2,3,3] / Argmin(2) / returns argmin indices along dim 2", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2., -5.},
					{0., -1., 3.},
					{7., -7., 7.},
				},
				{
					{8., -1., 0.},
					{5., 4., -3.},
					{1., -3., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Argmin(2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{2., 1., 1.},
				{1., 2., 1.},
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

		t.Run("scalar tensor / Argmin(-1) / returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Argmin(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "Argmin input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor / Argmin(0) / returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Argmin(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "Argmin input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1]) tensor / Argmin(1) / returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Argmin(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "Argmin input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([3,1]) tensor / Argmin(2) / returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			ten, err := tensor.Zeros([]int{3, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = ten.Argmin(2)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (2) being out of range")
			} else if err.Error() != "Argmin input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
