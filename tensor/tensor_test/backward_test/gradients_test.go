package backward_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestConcat(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("three grad-tracked inputs of shapes [5,1,3] [5,5,3] [5,2,3] / Concat along axis 1 then BackPropagate / each input gradient is all-ones matching its shape", func(t *testing.T) {
			x1, err := tensor.RandN([]int{5, 1, 3}, 0., 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.RandN([]int{5, 5, 3}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x3, err := tensor.RandU([]int{5, 2, 3}, -1., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Concat([]tensor.Tensor{x1, x2, x3}, 1)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act1 := x1.Gradient()
			act2 := x2.Gradient()
			act3 := x3.Gradient()

			exp1, err := tensor.Ones([]int{5, 1, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err := tensor.Ones([]int{5, 5, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			exp3, err := tensor.Ones([]int{5, 2, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act1, exp1)
			assertGradientEquals(t, act2, exp2)
			assertGradientEquals(t, act3, exp3)
		})

		t.Run("same grad-tracked [3,4] tensor used twice / Concat([x,x], axis 0) then BackPropagate / gradient of x is all-twos [3,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{3, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Concat([]tensor.Tensor{x, x}, 0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full([]int{3, 4}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("two grad-tracked [2,3] tensors / Concat along axis 0 then BackPropagate / each input gradient is all-ones [2,3]", func(t *testing.T) {
			x1, err := tensor.RandN([]int{2, 3}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.RandN([]int{2, 3}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Concat([]tensor.Tensor{x1, x2}, 0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act1 := x1.Gradient()
			act2 := x2.Gradient()

			exp1, err := tensor.Ones([]int{2, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err := tensor.Ones([]int{2, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act1, exp1)
			assertGradientEquals(t, act2, exp2)
		})

		t.Run("two grad-tracked [3,2] tensors / Concat along last axis (axis 1) then BackPropagate / each input gradient is all-ones [3,2]", func(t *testing.T) {
			x1, err := tensor.RandN([]int{3, 2}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.RandN([]int{3, 2}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Concat([]tensor.Tensor{x1, x2}, 1)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act1 := x1.Gradient()
			act2 := x2.Gradient()

			exp1, err := tensor.Ones([]int{3, 2}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err := tensor.Ones([]int{3, 2}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act1, exp1)
			assertGradientEquals(t, act2, exp2)
		})

		t.Run("two grad-tracked [2,3] tensors / Concat(axis 0) then SumAlong(0) then BackPropagate / each input gradient is all-ones [2,3]", func(t *testing.T) {
			x1, err := tensor.RandN([]int{2, 3}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.RandN([]int{2, 3}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			c, err := tensor.Concat([]tensor.Tensor{x1, x2}, 0)
			if err != nil {
				t.Fatal(err)
			}
			y, err := c.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act1 := x1.Gradient()
			act2 := x2.Gradient()

			exp1, err := tensor.Ones([]int{2, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err := tensor.Ones([]int{2, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act1, exp1)
			assertGradientEquals(t, act2, exp2)
		})

		// ============================== untracked paths ==============================

		t.Run("x1 grad-tracked, x2 untracked / Concat along axis 0 then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			x1, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Concat([]tensor.Tensor{x1, x2}, 0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("x1 untracked, x2 untracked / Concat along axis 0 then BackPropagate / y has nil gradient", func(t *testing.T) {
			x1, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Concat([]tensor.Tensor{x1, x2}, 0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestSlice(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

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

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [4,5] tensor sliced at two non-overlapping windows / Add slices then BackPropagate / gradient is 0 outside windows, 1 inside each", func(t *testing.T) {
			x, err := tensor.RandN([]int{4, 5}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			s1, err := x.Slice([]tensor.Range{{From: 0, To: 2}, {From: 0, To: 3}})
			if err != nil {
				t.Fatal(err)
			}
			s2, err := x.Slice([]tensor.Range{{From: 2, To: 4}, {From: 0, To: 3}})
			if err != nil {
				t.Fatal(err)
			}
			y, err := s1.Add(s2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Of([][]float64{
				{1., 1., 1., 0., 0.},
				{1., 1., 1., 0., 0.},
				{1., 1., 1., 0., 0.},
				{1., 1., 1., 0., 0.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [3,4] tensor / Slice(nil) then BackPropagate / gradient is all-ones [3,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{3, 4}, 0., 1., &tensor.Config{
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

			exp, err := tensor.Ones([]int{3, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [4,4,4] tensor / Slice([1:3],[1:3],[1:3]) then BackPropagate / gradient is 1 inside window, 0 outside", func(t *testing.T) {
			x, err := tensor.RandN([]int{4, 4, 4}, 0., 1., &tensor.Config{
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

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [4,6] tensor / Slice([1:3],[2:5]) then Reshape([6]) then Scale(2) then BackPropagate / gradient at sliced window is 2, elsewhere 0", func(t *testing.T) {
			x, err := tensor.RandN([]int{4, 6}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			s, err := x.Slice([]tensor.Range{{From: 1, To: 3}, {From: 2, To: 5}})
			if err != nil {
				t.Fatal(err)
			}
			r, err := s.Reshape([]int{6})
			if err != nil {
				t.Fatal(err)
			}
			y := r.Scale(2.)
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Of([][]float64{
				{0., 0., 0., 0., 0., 0.},
				{0., 0., 2., 2., 2., 0.},
				{0., 0., 2., 2., 2., 0.},
				{0., 0., 0., 0., 0., 0.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== untracked paths ==============================

		t.Run("x untracked / Slice(nil) then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
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

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestPatch(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("grad-tracked [4,5] x and grad-tracked [3,3] patch / Patch([1:4],[1:4]) then BackPropagate / x gradient is 1 outside patch window, 0 inside; patch gradient is all-ones", func(t *testing.T) {
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
			p, err := tensor.Of([][]float64{
				{-1., -2., -3.},
				{-4., -5., -6.},
				{-7., -8., -9.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Patch([]tensor.Range{{From: 1, To: 4}, {From: 1, To: 4}}, p)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			actx := x.Gradient()
			actp := p.Gradient()

			expx, err := tensor.Of([][]float64{
				{1., 1., 1., 1., 1.},
				{1., 0., 0., 0., 1.},
				{1., 0., 0., 0., 1.},
				{1., 0., 0., 0., 1.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expp, err := tensor.Of([][]float64{
				{1., 1., 1.},
				{1., 1., 1.},
				{1., 1., 1.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, actx, expx)
			assertGradientEquals(t, actp, expp)
		})

		t.Run("grad-tracked [4,4,4] base and [2,2,2] patch / Patch([1:3],[1:3],[1:3]) then BackPropagate / base gradient is 1 outside window 0 inside, patch gradient is all-ones", func(t *testing.T) {
			x, err := tensor.RandN([]int{4, 4, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			p, err := tensor.RandN([]int{2, 2, 2}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Patch([]tensor.Range{{From: 1, To: 3}, {From: 1, To: 3}, {From: 1, To: 3}}, p)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			actx := x.Gradient()
			actp := p.Gradient()

			expx, err := tensor.Of([][][]float64{
				{
					{1., 1., 1., 1.},
					{1., 1., 1., 1.},
					{1., 1., 1., 1.},
					{1., 1., 1., 1.},
				},
				{
					{1., 1., 1., 1.},
					{1., 0., 0., 1.},
					{1., 0., 0., 1.},
					{1., 1., 1., 1.},
				},
				{
					{1., 1., 1., 1.},
					{1., 0., 0., 1.},
					{1., 0., 0., 1.},
					{1., 1., 1., 1.},
				},
				{
					{1., 1., 1., 1.},
					{1., 1., 1., 1.},
					{1., 1., 1., 1.},
					{1., 1., 1., 1.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expp, err := tensor.Ones([]int{2, 2, 2}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, actx, expx)
			assertGradientEquals(t, actp, expp)
		})

		t.Run("grad-tracked [4,5] base, [2,2] patch p1 and p2 / sequential Patch then BackPropagate / gradients flow through both patch levels", func(t *testing.T) {
			x, err := tensor.RandN([]int{4, 5}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			p1, err := tensor.RandN([]int{2, 2}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			p2, err := tensor.RandN([]int{2, 2}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			mid, err := x.Patch([]tensor.Range{{From: 0, To: 2}, {From: 0, To: 2}}, p1)
			if err != nil {
				t.Fatal(err)
			}
			y, err := mid.Patch([]tensor.Range{{From: 2, To: 4}, {From: 2, To: 4}}, p2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			actx := x.Gradient()
			actp1 := p1.Gradient()
			actp2 := p2.Gradient()

			expx, err := tensor.Of([][]float64{
				{0., 0., 1., 1., 1.},
				{0., 0., 1., 1., 1.},
				{1., 1., 0., 0., 1.},
				{1., 1., 0., 0., 1.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expp1, err := tensor.Ones([]int{2, 2}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expp2, err := tensor.Ones([]int{2, 2}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, actx, expx)
			assertGradientEquals(t, actp1, expp1)
			assertGradientEquals(t, actp2, expp2)
		})

		// ============================== untracked paths ==============================

		t.Run("x untracked, p grad-tracked / Patch(nil) then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			p, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Patch(nil, p)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("x grad-tracked, p untracked / Patch(nil) then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			p, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Patch(nil, p)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("x untracked, p untracked / Patch(nil) then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			p, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Patch(nil, p)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestTranspose(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("grad-tracked [3,4] tensor / Transpose then BackPropagate / gradient of x is all-ones [3,4]", func(t *testing.T) {
			x, err := tensor.RandU([]int{3, 4}, -1., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{3, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [3,4] tensor / Transpose().Transpose() then BackPropagate / gradient of x is all-ones [3,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{3, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			t1, err := x.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			y, err := t1.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{3, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [3,3] identity matrix x / x.MatMul(x.Transpose()) then BackPropagate / gradient of x is all-twos [3,3]", func(t *testing.T) {
			x, err := tensor.Eye(3, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			xt, err := x.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			y, err := x.MatMul(xt)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full([]int{3, 3}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [2,4] tensor / Transpose then SumAlong(0) then BackPropagate / gradient of x is all-ones [2,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{2, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			xt, err := x.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			y, err := xt.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{2, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== untracked paths ==============================

		t.Run("x untracked / Transpose then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros([]int{1, 1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Transpose()
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestReshape(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("grad-tracked [3,4] tensor / Reshape([6,2]) then BackPropagate / gradient of x is all-ones [3,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{3, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Reshape([]int{6, 2})
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{3, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [3,4] tensor / Reshape([2,6]) then Scale(3) then Reshape([3,4]) then BackPropagate / gradient of x is all-threes [3,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{3, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			r1, err := x.Reshape([]int{2, 6})
			if err != nil {
				t.Fatal(err)
			}
			s := r1.Scale(3.)
			y, err := s.Reshape([]int{3, 4})
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full([]int{3, 4}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [2,2,2,2] tensor / Reshape([4,4]) then BackPropagate / gradient of x is all-ones [2,2,2,2]", func(t *testing.T) {
			x, err := tensor.RandN([]int{2, 2, 2, 2}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Reshape([]int{4, 4})
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{2, 2, 2, 2}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [3,4] tensor / Reshape([4,3]) then SumAlong(0) then BackPropagate / gradient of x is all-ones [3,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{3, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			r, err := x.Reshape([]int{4, 3})
			if err != nil {
				t.Fatal(err)
			}
			y, err := r.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{3, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== untracked paths ==============================

		t.Run("x untracked / Reshape(nil) then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Reshape(nil)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestUnsqueeze(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("grad-tracked [3,4] tensor / UnSqueeze(1) then BackPropagate / gradient of x is all-ones [3,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{3, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.UnSqueeze(1)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{3, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [3,4] tensor / UnSqueeze(0) then UnSqueeze(2) then BackPropagate / gradient of x is all-ones [3,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{3, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			u1, err := x.UnSqueeze(0)
			if err != nil {
				t.Fatal(err)
			}
			y, err := u1.UnSqueeze(2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{3, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [4] tensor / UnSqueeze(0) then Broadcast([3,4]) then SumAlong(0) then BackPropagate / gradient of x is all-ones [4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			u, err := x.UnSqueeze(0)
			if err != nil {
				t.Fatal(err)
			}
			b, err := u.Broadcast([]int{3, 4})
			if err != nil {
				t.Fatal(err)
			}
			y, err := b.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== untracked paths ==============================

		t.Run("x untracked / UnSqueeze(0) then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.UnSqueeze(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestSqueeze(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("grad-tracked [3,1,4] tensor / Squeeze(1) then BackPropagate / gradient of x is all-ones [3,1,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{3, 1, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Squeeze(1)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{3, 1, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [3,4] tensor / UnSqueeze(0) then Squeeze(0) then BackPropagate / gradient of x is all-ones [3,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{3, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			u, err := x.UnSqueeze(0)
			if err != nil {
				t.Fatal(err)
			}
			y, err := u.Squeeze(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{3, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [3,1,4] tensor / Squeeze(1) then Scale(2) then BackPropagate / gradient of x is all-twos [3,1,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{3, 1, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			s, err := x.Squeeze(1)
			if err != nil {
				t.Fatal(err)
			}
			y := s.Scale(2.)
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full([]int{3, 1, 4}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== untracked paths ==============================

		t.Run("x untracked / Squeeze(0) then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Squeeze(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestFlatten(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("grad-tracked [3,4] tensor / Flatten(0) then BackPropagate / gradient of x is all-ones [3,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{3, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Flatten(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{3, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [2,2,2,2] tensor / Flatten(0) then BackPropagate / gradient of x is all-ones [2,2,2,2]", func(t *testing.T) {
			x, err := tensor.RandN([]int{2, 2, 2, 2}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Flatten(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{2, 2, 2, 2}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [2,3,4] tensor / Flatten(1) then MatMul(Ones([12,5]) untracked) then BackPropagate / gradient of x is all-fives [2,3,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{2, 3, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			f, err := x.Flatten(1)
			if err != nil {
				t.Fatal(err)
			}

			w, err := tensor.Ones([]int{12, 5}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := f.MatMul(w)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full([]int{2, 3, 4}, 5., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== untracked paths ==============================

		t.Run("x untracked / Flatten(0) then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Flatten(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestBroadcast(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("grad-tracked [3,1,4] tensor / Broadcast([6,5,3,3,4]) then BackPropagate / gradient of x is all-ones [3,1,4]", func(t *testing.T) {
			x, err := tensor.RandU([]int{3, 1, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Broadcast([]int{6, 5, 3, 3, 4})
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{3, 1, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [1,1] tensor / Broadcast([4,4]) then BackPropagate / gradient of x is Full([1,1], 1.)", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 1}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Broadcast([]int{4, 4})
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full([]int{1, 1}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [1,4] tensor / Broadcast([3,4]) then Mul(untracked Ones([3,4])) then SumAlong(0) then BackPropagate / gradient of x is all-ones [1,4]", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			b, err := x.Broadcast([]int{3, 4})
			if err != nil {
				t.Fatal(err)
			}

			other, err := tensor.Ones([]int{3, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			m, err := b.Mul(other)
			if err != nil {
				t.Fatal(err)
			}

			y, err := m.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{1, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== untracked paths ==============================

		t.Run("x untracked / Broadcast(nil) then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Broadcast(nil)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestSum(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Of([2,4,4]) grad-tracked / SumAlong(1) then BackPropagate / gradient of x is all-ones [2,4,4]", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
					{4., 4., 4., 4.},
				},
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
					{4., 4., 4., 4.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.SumAlong(1)
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
					{1., 1., 1., 1.},
					{1., 1., 1., 1.},
					{1., 1., 1., 1.},
					{1., 1., 1., 1.},
				},
				{
					{1., 1., 1., 1.},
					{1., 1., 1., 1.},
					{1., 1., 1., 1.},
					{1., 1., 1., 1.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [4,3] tensor / SumAlong(0) then BackPropagate / gradient of x is all-ones [4,3]", func(t *testing.T) {
			x, err := tensor.RandN([]int{4, 3}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{4, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [2,3,4] tensor / SumAlong(2) then BackPropagate / gradient of x is all-ones [2,3,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{2, 3, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.SumAlong(2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Ones([]int{2, 3, 4}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [3,4] tensor / SumAlong(0) then Scale(2) then SumAlong(0) then BackPropagate / gradient of x is all-twos [3,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{3, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			s1, err := x.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			s2 := s1.Scale(2.)
			y, err := s2.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full([]int{3, 4}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== untracked paths ==============================

		t.Run("x untracked / SumAlong(0) then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestMax(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Of([2,4,4]) grad-tracked / MaxAlong(1) then BackPropagate / gradient concentrates on max row", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
					{4., 4., 4., 4.},
				},
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
					{4., 4., 4., 4.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MaxAlong(1)
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
					{1., 1., 1., 1.},
				},
				{
					{0., 0., 0., 0.},
					{0., 0., 0., 0.},
					{0., 0., 0., 0.},
					{1., 1., 1., 1.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			assertGradientEquals(t, act, exp)
		})

		t.Run("x untracked / MaxAlong(0) then BackPropagate / result has no gradient", func(t *testing.T) {
			x, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MaxAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestMin(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("3D tensor with GradTrack / MinAlong(1) then BackPropagate / gradient equals 1 at minimum row, 0 elsewhere", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
					{4., 4., 4., 4.},
				},
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
					{4., 4., 4., 4.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MinAlong(1)
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
					{1., 1., 1., 1.},
					{0., 0., 0., 0.},
					{0., 0., 0., 0.},
					{0., 0., 0., 0.},
				},
				{
					{1., 1., 1., 1.},
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

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [1,4] tensor with three min-ties / MinAlong(1) then BackPropagate / gradient is 1 at each tied min position", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{5., 3., 3., 3.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MinAlong(1)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()
			exp, err := tensor.Of([][]float64{
				{0., 1., 1., 1.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [3,2,2] tensor / MinAlong(0) then BackPropagate / gradient is 1 at batch-min positions, 0 elsewhere", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{2., 2.},
					{2., 2.},
				},
				{
					{1., 1.},
					{1., 1.},
				},
				{
					{3., 3.},
					{3., 3.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MinAlong(0)
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
					{0., 0.},
					{0., 0.},
				},
				{
					{1., 1.},
					{1., 1.},
				},
				{
					{0., 0.},
					{0., 0.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== untracked paths ==============================

		t.Run("scalar tensor without GradTrack / MinAlong(0) then BackPropagate / result gradient is nil", func(t *testing.T) {
			x, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MinAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestAvg(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("grad-tracked [2][4][4] tensor / AvgAlong(1) then BackPropagate / gradient is 0.25 everywhere", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
					{4., 4., 4., 4.},
				},
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
					{4., 4., 4., 4.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(1)
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
					{0.25, 0.25, 0.25, 0.25},
					{0.25, 0.25, 0.25, 0.25},
					{0.25, 0.25, 0.25, 0.25},
					{0.25, 0.25, 0.25, 0.25},
				},
				{
					{0.25, 0.25, 0.25, 0.25},
					{0.25, 0.25, 0.25, 0.25},
					{0.25, 0.25, 0.25, 0.25},
					{0.25, 0.25, 0.25, 0.25},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [2,3,4] tensor / AvgAlong(2) then BackPropagate / gradient is 0.25 everywhere [2,3,4]", func(t *testing.T) {
			x, err := tensor.RandN([]int{2, 3, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full([]int{2, 3, 4}, 0.25, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [3,4] tensor / AvgAlong(0) then Add(untracked) then SumAlong(0) then BackPropagate / gradient of x is 1/3 everywhere", func(t *testing.T) {
			x, err := tensor.RandN([]int{3, 4}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			c, err := tensor.Full([]int{1, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			a, err := x.AvgAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			s, err := a.Add(c)
			if err != nil {
				t.Fatal(err)
			}
			y, err := s.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full([]int{3, 4}, 1./3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== untracked paths ==============================

		t.Run("untracked [1] tensor / AvgAlong(0) then BackPropagate / result has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestVar(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Ones([5,1,3]) grad-tracked / VarAlong(1) then BackPropagate / gradient of x is all-zeros [5,1,3]", func(t *testing.T) {
			x, err := tensor.Ones([]int{5, 1, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(1)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Zeros([]int{5, 1, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [2][3][4] tensor with rows 1,2,3 / VarAlong(1) then BackPropagate / gradient reflects variance derivative", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
				},
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(1)
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
					{-1., -1., -1., -1.},
					{0., 0., 0., 0.},
					{1., 1., 1., 1.},
				},
				{
					{-1., -1., -1., -1.},
					{0., 0., 0., 0.},
					{1., 1., 1., 1.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [2,4,3] tensor with rows 1/2/3/4 / VarAlong(1) then BackPropagate / gradient reflects variance derivative analytically", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1., 1., 1.},
					{2., 2., 2.},
					{3., 3., 3.},
					{4., 4., 4.},
				},
				{
					{1., 1., 1.},
					{2., 2., 2.},
					{3., 3., 3.},
					{4., 4., 4.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(1)
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
					{-1., -1., -1.},
					{-1. / 3., -1. / 3., -1. / 3.},
					{1. / 3., 1. / 3., 1. / 3.},
					{1., 1., 1.},
				},
				{
					{-1., -1., -1.},
					{-1. / 3., -1. / 3., -1. / 3.},
					{1. / 3., 1. / 3., 1. / 3.},
					{1., 1., 1.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [2,1,3] tensor / VarAlong(1) on single-element groups then BackPropagate / gradient is all-zeros [2,1,3]", func(t *testing.T) {
			x, err := tensor.RandN([]int{2, 1, 3}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(1)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Zeros([]int{2, 1, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== untracked paths ==============================

		t.Run("untracked [1] tensor / VarAlong(0) then BackPropagate / result has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestStd(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Ones([5,1,3]) gradient-tracked / StdAlong(1) then BackPropagate / gradient is zeros", func(t *testing.T) {
			x, err := tensor.Ones([]int{5, 1, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(1)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()
			exp, err := tensor.Zeros([]int{5, 1, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			assertGradientEquals(t, act, exp)
		})

		t.Run("[2,3,4] tensor with values 1/2/3 along axis 1 / StdAlong(1) then BackPropagate / gradient is -0.5/0/0.5 pattern", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
				},
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(1)
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
					{-0.5, -0.5, -0.5, -0.5},
					{0., 0., 0., 0.},
					{0.5, 0.5, 0.5, 0.5},
				},
				{
					{-0.5, -0.5, -0.5, -0.5},
					{0., 0., 0., 0.},
					{0.5, 0.5, 0.5, 0.5},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [2,3,4] tensor / StdAlong(1) then Scale(2) then MeanAlong(0) then BackPropagate / gradient is -0.5/0/0.5 pattern", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
				},
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			s, err := x.StdAlong(1)
			if err != nil {
				t.Fatal(err)
			}
			sc := s.Scale(2.)
			y, err := sc.MeanAlong(0)
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
					{-0.5, -0.5, -0.5, -0.5},
					{0., 0., 0., 0.},
					{0.5, 0.5, 0.5, 0.5},
				},
				{
					{-0.5, -0.5, -0.5, -0.5},
					{0., 0., 0., 0.},
					{0.5, 0.5, 0.5, 0.5},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== untracked paths ==============================

		t.Run("single-element tensor without grad tracking / StdAlong(0) then BackPropagate / output gradient is nil", func(t *testing.T) {
			x, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestMean(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("grad-tracked [2][4][4] tensor / MeanAlong(1) then BackPropagate / gradient is 0.25 everywhere", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
					{4., 4., 4., 4.},
				},
				{
					{1., 1., 1., 1.},
					{2., 2., 2., 2.},
					{3., 3., 3., 3.},
					{4., 4., 4., 4.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MeanAlong(1)
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
					{0.25, 0.25, 0.25, 0.25},
					{0.25, 0.25, 0.25, 0.25},
					{0.25, 0.25, 0.25, 0.25},
					{0.25, 0.25, 0.25, 0.25},
				},
				{
					{0.25, 0.25, 0.25, 0.25},
					{0.25, 0.25, 0.25, 0.25},
					{0.25, 0.25, 0.25, 0.25},
					{0.25, 0.25, 0.25, 0.25},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [4,3] tensor / MeanAlong(0) then BackPropagate / gradient is 0.25 everywhere [4,3]", func(t *testing.T) {
			x, err := tensor.RandN([]int{4, 3}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MeanAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full([]int{4, 3}, 0.25, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("grad-tracked [4,3] tensor / MeanAlong(0) then Scale(2) then MeanAlong(0) then BackPropagate / gradient is 1/6 everywhere [4,3]", func(t *testing.T) {
			x, err := tensor.RandN([]int{4, 3}, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			m, err := x.MeanAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			sc := m.Scale(2.)
			y, err := sc.MeanAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full([]int{4, 3}, 1./6., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		// ============================== untracked paths ==============================

		t.Run("untracked [1] tensor / MeanAlong(0) then BackPropagate / result has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MeanAlong(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestScale(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 2) grad-tracked / Scale(3) then BackPropagate / gradient of x is 3", func(t *testing.T) {
			x, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(3.)
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("x untracked / Scale(0) then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(0.)
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestPow(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 2) grad-tracked / Pow(3) then BackPropagate / gradient of x is 12", func(t *testing.T) {
			x, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(3.)
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full(nil, 12., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("x untracked / Pow(0) then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(0.)
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestExp(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 1) grad-tracked / Exp then BackPropagate / gradient of x is e", func(t *testing.T) {
			x, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Exp()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full(nil, math.E, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("x untracked / Exp then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Exp()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestLog(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 2) grad-tracked / Log then BackPropagate / gradient of x is 0.5", func(t *testing.T) {
			x, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full(nil, 0.5, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("x untracked / Log then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestSin(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, π/3) grad-tracked / Sin then BackPropagate / gradient of x is cos(π/3)", func(t *testing.T) {
			x, err := tensor.Full(nil, math.Pi/3, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sin()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full(nil, math.Cos(math.Pi/3), &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("x untracked / Sin then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sin()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestCos(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, π/6) grad-tracked / Cos then BackPropagate / gradient of x is -sin(π/6)", func(t *testing.T) {
			x, err := tensor.Full(nil, math.Pi/6, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cos()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full(nil, -math.Sin(math.Pi/6), &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("x untracked / Cos then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cos()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestTan(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 0) grad-tracked / Tan then BackPropagate / gradient of x is sec²(0) = 1", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tan()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			exp, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, act, exp)
		})

		t.Run("x untracked / Tan then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tan()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestSinh(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 1) grad-tracked / Sinh then BackPropagate / gradient of x is cosh(1)", func(t *testing.T) {
			x, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sinh()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()
			c := (math.E*math.E + 1) / (2 * math.E)

			if val, err := act.At(); err != nil {
				t.Fatal(err)
			} else if !(c-1e-10 < val && val < c+1e-10) {
				t.Fatal("expected tensors to be equal")
			} else if act.Gradient() != nil {
				t.Fatal("expected gradient of gradient to be nil (gradient tensors should not track their own gradients)")
			}
		})

		t.Run("x untracked / Sinh then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sinh()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestCosh(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 1) grad-tracked / Cosh then BackPropagate / gradient of x is sinh(1)", func(t *testing.T) {
			x, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cosh()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()
			c := (math.E*math.E - 1) / (2 * math.E)

			if val, err := act.At(); err != nil {
				t.Fatal(err)
			} else if !(c-1e-10 < val && val < c+1e-10) {
				t.Fatal("expected tensors to be equal")
			} else if act.Gradient() != nil {
				t.Fatal("expected gradient of gradient to be nil (gradient tensors should not track their own gradients)")
			}
		})

		t.Run("x untracked / Cosh then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cosh()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestTanh(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 0) grad-tracked / Tanh then BackPropagate / gradient of x is sech²(0) = 1", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tanh()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			act := x.Gradient()

			if val, err := act.At(); err != nil {
				t.Fatal(err)
			} else if !(1.-1e-10 < val && val < 1.+1e-10) {
				t.Fatal("expected tensors to be equal")
			} else if act.Gradient() != nil {
				t.Fatal("expected gradient of gradient to be nil (gradient tensors should not track their own gradients)")
			}
		})

		t.Run("x untracked / Tanh then BackPropagate / y has nil gradient", func(t *testing.T) {
			x, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tanh()
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestElMax(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked / ElMax then BackPropagate / gradient of a is 1, gradient of b is 0", func(t *testing.T) {
			a, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.ElMax(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			acta := a.Gradient()
			actb := b.Gradient()

			expa, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expb, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, acta, expa)
			assertGradientEquals(t, actb, expb)
		})

		t.Run("Full(nil, 2) and Full(nil, 2) both grad-tracked / ElMax then BackPropagate / gradient of a is 0.5, gradient of b is 0.5", func(t *testing.T) {
			a, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.ElMax(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			acta := a.Gradient()
			actb := b.Gradient()

			expa, err := tensor.Full(nil, 0.5, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expb, err := tensor.Full(nil, 0.5, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, acta, expa)
			assertGradientEquals(t, actb, expb)
		})

		t.Run("a untracked, b grad-tracked / ElMax then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.ElMax(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a grad-tracked, b untracked / ElMax then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.ElMax(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a untracked, b untracked / ElMax then BackPropagate / y has nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.ElMax(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestElMin(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked / ElMin then BackPropagate / gradient of a is 0, gradient of b is 1", func(t *testing.T) {
			a, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.ElMin(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			acta := a.Gradient()
			actb := b.Gradient()

			expa, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expb, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, acta, expa)
			assertGradientEquals(t, actb, expb)
		})

		t.Run("Full(nil, 2) and Full(nil, 2) both grad-tracked / ElMin then BackPropagate / gradient of a is 0.5, gradient of b is 0.5", func(t *testing.T) {
			a, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.ElMin(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			acta := a.Gradient()
			actb := b.Gradient()

			expa, err := tensor.Full(nil, 0.5, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expb, err := tensor.Full(nil, 0.5, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, acta, expa)
			assertGradientEquals(t, actb, expb)
		})

		t.Run("a untracked, b grad-tracked / ElMin then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.ElMin(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a grad-tracked, b untracked / ElMin then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.ElMin(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a untracked, b untracked / ElMin then BackPropagate / y has nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.ElMin(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestAdd(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked / Add then BackPropagate / gradient of a is 1, gradient of b is 1", func(t *testing.T) {
			a, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Add(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			acta := a.Gradient()
			actb := b.Gradient()

			expa, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expb, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, acta, expa)
			assertGradientEquals(t, actb, expb)
		})

		t.Run("a untracked, b grad-tracked / Add then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Add(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a grad-tracked, b untracked / Add then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Add(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a untracked, b untracked / Add then BackPropagate / y has nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Add(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestSub(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked / Sub then BackPropagate / gradient of a is 1, gradient of b is -1", func(t *testing.T) {
			a, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Sub(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			acta := a.Gradient()
			actb := b.Gradient()

			expa, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expb, err := tensor.Full(nil, -1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, acta, expa)
			assertGradientEquals(t, actb, expb)
		})

		t.Run("a untracked, b grad-tracked / Sub then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Sub(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a grad-tracked, b untracked / Sub then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Sub(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a untracked, b untracked / Sub then BackPropagate / y has nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Sub(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestMul(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked / Mul then BackPropagate / gradient of a is 2, gradient of b is 3", func(t *testing.T) {
			a, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Mul(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			acta := a.Gradient()
			actb := b.Gradient()

			expa, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expb, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, acta, expa)
			assertGradientEquals(t, actb, expb)
		})

		t.Run("a untracked, b grad-tracked / Mul then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Mul(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a grad-tracked, b untracked / Mul then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Mul(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a untracked, b untracked / Mul then BackPropagate / y has nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Mul(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestDiv(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked / Div then BackPropagate / gradient of a is 0.5, gradient of b is -0.75", func(t *testing.T) {
			a, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Div(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			acta := a.Gradient()
			actb := b.Gradient()

			expa, err := tensor.Full(nil, 0.5, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expb, err := tensor.Full(nil, -0.75, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, acta, expa)
			assertGradientEquals(t, actb, expb)
		})

		t.Run("a untracked, b grad-tracked / Div then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Div(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a grad-tracked, b untracked / Div then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Div(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a untracked, b untracked / Div then BackPropagate / y has nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Div(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestDot(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full([2], 2) and Full([2], 3) both grad-tracked / Dot then BackPropagate / gradient of a is Full([2], 3), gradient of b is Full([2], 2)", func(t *testing.T) {
			a, err := tensor.Full([]int{2}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full([]int{2}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Dot(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			acta := a.Gradient()
			actb := b.Gradient()

			expa, err := tensor.Full([]int{2}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expb, err := tensor.Full([]int{2}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, acta, expa)
			assertGradientEquals(t, actb, expb)
		})

		t.Run("a untracked, b grad-tracked / Dot then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Dot(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a grad-tracked, b untracked / Dot then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Dot(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a untracked, b untracked / Dot then BackPropagate / y has nil gradient", func(t *testing.T) {
			a, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Zeros([]int{1}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.Dot(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

func TestMatmul(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full([2,3], 2) and Full([3,2], 3) both grad-tracked / MatMul then BackPropagate / gradient of a is Full([2,3], 6), gradient of b is Full([3,2], 4)", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 3}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full([]int{3, 2}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.MatMul(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			acta := a.Gradient()
			actb := b.Gradient()

			expa, err := tensor.Full([]int{2, 3}, 6., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			expb, err := tensor.Full([]int{3, 2}, 4., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			assertGradientEquals(t, acta, expa)
			assertGradientEquals(t, actb, expb)
		})

		t.Run("a untracked, b grad-tracked / MatMul then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Eye(1, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Eye(1, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.MatMul(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a grad-tracked, b untracked / MatMul then BackPropagate / y has non-nil gradient", func(t *testing.T) {
			a, err := tensor.Eye(1, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Eye(1, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.MatMul(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() == nil {
				t.Fatal("expected gradient not to be nil")
			}
		})

		t.Run("a untracked, b untracked / MatMul then BackPropagate / y has nil gradient", func(t *testing.T) {
			a, err := tensor.Eye(1, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Eye(1, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := a.MatMul(b)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})
	})
}

/* ----- helpers ----- */

func assertGradientEquals(t *testing.T, act, exp tensor.Tensor) {
	t.Helper()

	if eq, err := act.Equals(exp); err != nil {
		t.Fatal(err)
	} else if !eq {
		t.Fatal("expected tensors to be equal")
	}

	if act.Gradient() != nil {
		t.Fatal("expected gradient of gradient to be nil (gradient tensors should not track their own gradients)")
	}
}
