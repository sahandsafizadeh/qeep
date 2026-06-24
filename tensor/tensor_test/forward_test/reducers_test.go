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

			if val := ten.Sum(); !(9.-1e-10 < val && val < 9.+1e-10) {
				t.Fatalf("expected (9) as the sum value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [6, 4] / Sum() / returns 10", func(t *testing.T) {
			ten, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Sum(); !(10.-1e-10 < val && val < 10.+1e-10) {
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

			if val := ten.Sum(); !(36.-1e-10 < val && val < 36.+1e-10) {
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

			if val := ten.Max(); !(9.-1e-10 < val && val < 9.+1e-10) {
				t.Fatalf("expected (9) as the max value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [6, 4] / Max() / returns 6", func(t *testing.T) {
			ten, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Max(); !(6.-1e-10 < val && val < 6.+1e-10) {
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

			if val := ten.Max(); !(9.-1e-10 < val && val < 9.+1e-10) {
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

			if val := ten.Min(); !(9.-1e-10 < val && val < 9.+1e-10) {
				t.Fatalf("expected (9) as the min value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [6, 4] / Min() / returns 4", func(t *testing.T) {
			ten, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Min(); !(4.-1e-10 < val && val < 4.+1e-10) {
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

			if val := ten.Min(); !(-5.-1e-10 < val && val < -5.+1e-10) {
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

			if val := ten.Avg(); !(9.-1e-10 < val && val < 9.+1e-10) {
				t.Fatalf("expected (9) as the avg value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [6, 4] / Avg() / returns 5", func(t *testing.T) {
			ten, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Avg(); !(5.-1e-10 < val && val < 5.+1e-10) {
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

			if val := ten.Avg(); !(3.-1e-10 < val && val < 3.+1e-10) {
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

			if val := ten.Var(); !(0.-1e-10 < val && val < 0.+1e-10) {
				t.Fatalf("expected (0) as the var value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [-sqrt2, sqrt2] / Var() / returns 4", func(t *testing.T) {
			ten, err := tensor.Of([]float64{-math.Sqrt2, math.Sqrt2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Var(); !(4.-1e-10 < val && val < 4.+1e-10) {
				t.Fatalf("expected (4) as the var value of tensor, got (%f)", val)
			}
		})

		t.Run("3D tensor shape [1,3,1] / Var() / returns 4", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{{{-2.}, {0.}, {2.}}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Var(); !(4.-1e-10 < val && val < 4.+1e-10) {
				t.Fatalf("expected (4) as the var value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [5, 5, 5] / Var() / returns 0", func(t *testing.T) {
			ten, err := tensor.Of([]float64{5., 5., 5.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Var(); !(0.-1e-10 < val && val < 0.+1e-10) {
				t.Fatalf("expected (0) as the var value of tensor, got (%f)", val)
			}
		})

		t.Run("2D tensor shape [2,4] / Var() / returns 6", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{1., 2., 3., 4.},
				{5., 6., 7., 8.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Var(); !(6.-1e-10 < val && val < 6.+1e-10) {
				t.Fatalf("expected (6) as the var value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [-3, 0, 3] / Var() / returns 9", func(t *testing.T) {
			ten, err := tensor.Of([]float64{-3., 0., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Var(); !(9.-1e-10 < val && val < 9.+1e-10) {
				t.Fatalf("expected (9) as the var value of tensor, got (%f)", val)
			}
		})

		t.Run("3D tensor shape [2,2,3] / Var() / returns 13", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2., 3.},
					{4., 5., 6.},
				},
				{
					{7., 8., 9.},
					{10., 11., 12.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Var(); !(13.-1e-10 < val && val < 13.+1e-10) {
				t.Fatalf("expected (13) as the var value of tensor, got (%f)", val)
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

			if val := ten.Std(); !(0.-1e-10 < val && val < 0.+1e-10) {
				t.Fatalf("expected (0) as the std value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [-sqrt2, sqrt2] / Std() / returns 2", func(t *testing.T) {
			ten, err := tensor.Of([]float64{-math.Sqrt2, math.Sqrt2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Std(); !(2.-1e-10 < val && val < 2.+1e-10) {
				t.Fatalf("expected (2) as the std value of tensor, got (%f)", val)
			}
		})

		t.Run("3D tensor shape [1,3,1] / Std() / returns 2", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{{{-2.}, {0.}, {2.}}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Std(); !(2.-1e-10 < val && val < 2.+1e-10) {
				t.Fatalf("expected (2) as the std value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [7, 7, 7] / Std() / returns 0", func(t *testing.T) {
			ten, err := tensor.Of([]float64{7., 7., 7.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Std(); !(0.-1e-10 < val && val < 0.+1e-10) {
				t.Fatalf("expected (0) as the std value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [-3, 0, 3] / Std() / returns 3", func(t *testing.T) {
			ten, err := tensor.Of([]float64{-3., 0., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Std(); !(3.-1e-10 < val && val < 3.+1e-10) {
				t.Fatalf("expected (3) as the std value of tensor, got (%f)", val)
			}
		})

		t.Run("2D tensor shape [2,4] / Std() / returns sqrt(6)", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{1., 2., 3., 4.},
				{5., 6., 7., 8.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Std(); !(math.Sqrt(6)-1e-10 < val && val < math.Sqrt(6)+1e-10) {
				t.Fatalf("expected (sqrt(6)) as the std value of tensor, got (%f)", val)
			}
		})

		t.Run("3D tensor shape [2,2,3] / Std() / returns sqrt(13)", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 2., 3.},
					{4., 5., 6.},
				},
				{
					{7., 8., 9.},
					{10., 11., 12.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Std(); !(math.Sqrt(13)-1e-10 < val && val < math.Sqrt(13)+1e-10) {
				t.Fatalf("expected (sqrt(13)) as the std value of tensor, got (%f)", val)
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

			if val := ten.Mean(); !(9.-1e-10 < val && val < 9.+1e-10) {
				t.Fatalf("expected (9) as the mean value of tensor, got (%f)", val)
			}
		})

		t.Run("1D tensor [6, 4] / Mean() / returns 5", func(t *testing.T) {
			ten, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val := ten.Mean(); !(5.-1e-10 < val && val < 5.+1e-10) {
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

			if val := ten.Mean(); !(3.-1e-10 < val && val < 3.+1e-10) {
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

		t.Run("4D tensor shape [2,2,2,2] / SumAlong(1) / returns [2,2,2] tensor", func(t *testing.T) {
			ten, err := tensor.Of([][][][]float64{
				{
					{
						{1., 2.},
						{3., 4.},
					},
					{
						{5., 6.},
						{7., 8.},
					},
				},
				{
					{
						{9., 10.},
						{11., 12.},
					},
					{
						{13., 14.},
						{15., 16.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.SumAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{6., 8.},
					{10., 12.},
				},
				{
					{22., 24.},
					{26., 28.},
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

		t.Run("4D tensor shape [2,3,2,2] / SumAlong(3) / returns [2,3,2] tensor", func(t *testing.T) {
			ten, err := tensor.Of([][][][]float64{
				{
					{
						{1., 2.},
						{3., 4.},
					},
					{
						{5., 6.},
						{7., 8.},
					},
					{
						{9., 10.},
						{11., 12.},
					},
				},
				{
					{
						{-1., -2.},
						{-3., -4.},
					},
					{
						{-5., -6.},
						{-7., -8.},
					},
					{
						{-9., -10.},
						{-11., -12.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.SumAlong(3)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{3., 7.},
					{11., 15.},
					{19., 23.},
				},
				{
					{-3., -7.},
					{-11., -15.},
					{-19., -23.},
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

		t.Run("Of([2,2,2,2]) tensor / AvgAlong(1) / returns Of([2,2,2])", func(t *testing.T) {
			ten, err := tensor.Of([][][][]float64{
				{
					{
						{1., 2.},
						{3., 4.},
					},
					{
						{5., 6.},
						{7., 8.},
					},
				},
				{
					{
						{9., 10.},
						{11., 12.},
					},
					{
						{13., 14.},
						{15., 16.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.AvgAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{3., 4.},
					{5., 6.},
				},
				{
					{11., 12.},
					{13., 14.},
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

		t.Run("Of([2,3,2,2]) tensor / AvgAlong(3) / returns Of([2,3,2])", func(t *testing.T) {
			ten, err := tensor.Of([][][][]float64{
				{
					{
						{1., 2.},
						{3., 4.},
					},
					{
						{5., 6.},
						{7., 8.},
					},
					{
						{9., 10.},
						{11., 12.},
					},
				},
				{
					{
						{-1., -2.},
						{-3., -4.},
					},
					{
						{-5., -6.},
						{-7., -8.},
					},
					{
						{-9., -10.},
						{-11., -12.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.AvgAlong(3)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{1.5, 3.5},
					{5.5, 7.5},
					{9.5, 11.5},
				},
				{
					{-1.5, -3.5},
					{-5.5, -7.5},
					{-9.5, -11.5},
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

		t.Run("tensor shape [3] / VarAlong(0) / returns scalar 9", func(t *testing.T) {
			ten, err := tensor.Of([]float64{-3., 0., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.VarAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("tensor shape [3,3] / VarAlong(0) / returns [3] tensor with all 9", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
				{7., 8., 9.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.VarAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{9., 9., 9.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("tensor shape [3,2,2] / VarAlong(0) / returns [2,2] tensor with all 1", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 4.},
					{7., 10.},
				},
				{
					{2., 5.},
					{8., 11.},
				},
				{
					{3., 6.},
					{9., 12.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.VarAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{1., 1.},
				{1., 1.},
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

		t.Run("tensor shape [3,2,2] / VarAlong(1) / returns [3,2] tensor with all 18", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 4.},
					{7., 10.},
				},
				{
					{2., 5.},
					{8., 11.},
				},
				{
					{3., 6.},
					{9., 12.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.VarAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{18., 18.},
				{18., 18.},
				{18., 18.},
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

		t.Run("tensor shape [3,2,2] / VarAlong(2) / returns [3,2] tensor with all 4.5", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 4.},
					{7., 10.},
				},
				{
					{2., 5.},
					{8., 11.},
				},
				{
					{3., 6.},
					{9., 12.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.VarAlong(2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{4.5, 4.5},
				{4.5, 4.5},
				{4.5, 4.5},
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

		t.Run("tensor shape [1,2,4,5] / VarAlong(3) / returns [1,2,4] tensor", func(t *testing.T) {
			ten, err := tensor.Of([][][][]float64{
				{
					{
						{0., 2., 4., 6., 8.},
						{0., 4., 8., 12., 16.},
						{0., 6., 12., 18., 24.},
						{0., 8., 16., 24., 32.},
					},
					{
						{0., 10., 20., 30., 40.},
						{0., 12., 24., 36., 48.},
						{0., 14., 28., 42., 56.},
						{0., 16., 32., 48., 64.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.VarAlong(3)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{10., 40., 90., 160.},
					{250., 360., 490., 640.},
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

		t.Run("tensor shape [3] / StdAlong(0) / returns scalar 3", func(t *testing.T) {
			ten, err := tensor.Of([]float64{-3., 0., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.StdAlong(0)
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

		t.Run("tensor shape [3,3] / StdAlong(0) / returns [3] tensor with all 3", func(t *testing.T) {
			ten, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
				{7., 8., 9.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.StdAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{3., 3., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("tensor shape [3,2,2] / StdAlong(0) / returns [2,2] tensor with all 1", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 4.},
					{7., 10.},
				},
				{
					{2., 5.},
					{8., 11.},
				},
				{
					{3., 6.},
					{9., 12.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.StdAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{1., 1.},
				{1., 1.},
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

		t.Run("tensor shape [3,2,2] / StdAlong(1) / returns [3,2] tensor with all sqrt(18)", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 4.},
					{7., 10.},
				},
				{
					{2., 5.},
					{8., 11.},
				},
				{
					{3., 6.},
					{9., 12.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.StdAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{math.Sqrt(18.), math.Sqrt(18.)},
				{math.Sqrt(18.), math.Sqrt(18.)},
				{math.Sqrt(18.), math.Sqrt(18.)},
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

		t.Run("tensor shape [3,2,2] / StdAlong(2) / returns [3,2] tensor with all sqrt(4.5)", func(t *testing.T) {
			ten, err := tensor.Of([][][]float64{
				{
					{1., 4.},
					{7., 10.},
				},
				{
					{2., 5.},
					{8., 11.},
				},
				{
					{3., 6.},
					{9., 12.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.StdAlong(2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{math.Sqrt(4.5), math.Sqrt(4.5)},
				{math.Sqrt(4.5), math.Sqrt(4.5)},
				{math.Sqrt(4.5), math.Sqrt(4.5)},
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

		t.Run("tensor shape [1,2,4,5] / StdAlong(3) / returns [1,2,4] tensor", func(t *testing.T) {
			ten, err := tensor.Of([][][][]float64{
				{
					{
						{1., 3., 5., 7., 9.},
						{2., 6., 10., 14., 18.},
						{3., 9., 15., 21., 27.},
						{4., 12., 20., 28., 36.},
					},
					{
						{5., 15., 25., 35., 45.},
						{6., 18., 30., 42., 54.},
						{7., 21., 35., 49., 63.},
						{8., 24., 40., 56., 72.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.StdAlong(3)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{math.Sqrt(10.), math.Sqrt(40.), math.Sqrt(90.), math.Sqrt(160.)},
					{math.Sqrt(250.), math.Sqrt(360.), math.Sqrt(490.), math.Sqrt(640.)},
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

		t.Run("Of([2,2,2,2]) tensor / MeanAlong(1) / returns Of([2,2,2])", func(t *testing.T) {
			ten, err := tensor.Of([][][][]float64{
				{
					{
						{1., 2.},
						{3., 4.},
					},
					{
						{5., 6.},
						{7., 8.},
					},
				},
				{
					{
						{9., 10.},
						{11., 12.},
					},
					{
						{13., 14.},
						{15., 16.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MeanAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{3., 4.},
					{5., 6.},
				},
				{
					{11., 12.},
					{13., 14.},
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

		t.Run("Of([2,3,2,2]) tensor / MeanAlong(3) / returns Of([2,3,2])", func(t *testing.T) {
			ten, err := tensor.Of([][][][]float64{
				{
					{
						{1., 2.},
						{3., 4.},
					},
					{
						{5., 6.},
						{7., 8.},
					},
					{
						{9., 10.},
						{11., 12.},
					},
				},
				{
					{
						{-1., -2.},
						{-3., -4.},
					},
					{
						{-5., -6.},
						{-7., -8.},
					},
					{
						{-9., -10.},
						{-11., -12.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.MeanAlong(3)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{1.5, 3.5},
					{5.5, 7.5},
					{9.5, 11.5},
				},
				{
					{-1.5, -3.5},
					{-5.5, -7.5},
					{-9.5, -11.5},
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

		t.Run("tensor shape [2,4,2,2] / Argmax(1) / returns argmax indices along dim 1", func(t *testing.T) {
			ten, err := tensor.Of([][][][]float64{
				{
					{
						{3., -1.},
						{0., 7.},
					},
					{
						{1., 5.},
						{6., -3.},
					},
					{
						{4., 2.},
						{-2., 1.},
					},
					{
						{2., 0.},
						{8., 4.},
					},
				},
				{
					{
						{2., -4.},
						{1., 6.},
					},
					{
						{9., 0.},
						{-5., 2.},
					},
					{
						{-1., 3.},
						{4., -3.},
					},
					{
						{5., -2.},
						{0., 8.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Argmax(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{2., 1.},
					{3., 0.},
				},
				{
					{1., 2.},
					{2., 3.},
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

		t.Run("tensor shape [2,4,2,2] / Argmin(1) / returns argmin indices along dim 1", func(t *testing.T) {
			ten, err := tensor.Of([][][][]float64{
				{
					{
						{3., -1.},
						{0., 7.},
					},
					{
						{1., 5.},
						{6., -3.},
					},
					{
						{4., 2.},
						{-2., 1.},
					},
					{
						{2., 0.},
						{8., 4.},
					},
				},
				{
					{
						{2., -4.},
						{1., 6.},
					},
					{
						{9., 0.},
						{-5., 2.},
					},
					{
						{-1., 3.},
						{4., -3.},
					},
					{
						{5., -2.},
						{0., 8.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := ten.Argmin(1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{1., 0.},
					{2., 1.},
				},
				{
					{2., 0.},
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
