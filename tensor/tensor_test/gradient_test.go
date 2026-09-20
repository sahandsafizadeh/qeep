package tensor_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

// import (
// 	"math"
// 	"testing"

// 	"github.com/sahandsafizadeh/qeep/tensor"
// )

// func TestPatch(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("grad-tracked [4,5] x and grad-tracked [3,3] patch / Patch([1:4],[1:4]) then BackPropagate / x gradient is 1 outside patch window, 0 inside; patch gradient is all-ones", func(t *testing.T) {
// 			x, err := tensor.Of([][]float64{
// 				{0., 1., 2., 3., 4.},
// 				{5., 6., 7., 8., 9.},
// 				{4., 3., 2., 1., 0.},
// 				{9., 8., 7., 6., 5.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			p, err := tensor.Of([][]float64{
// 				{-1., -2., -3.},
// 				{-4., -5., -6.},
// 				{-7., -8., -9.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := x.Patch([]tensor.Range{{From: 1, To: 4}, {From: 1, To: 4}}, p)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			actx := x.Gradient()
// 			actp := p.Gradient()

// 			expx, err := tensor.Of([][]float64{
// 				{1., 1., 1., 1., 1.},
// 				{1., 0., 0., 0., 1.},
// 				{1., 0., 0., 0., 1.},
// 				{1., 0., 0., 0., 1.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expp, err := tensor.Of([][]float64{
// 				{1., 1., 1.},
// 				{1., 1., 1.},
// 				{1., 1., 1.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, actx, expx)
// 			assertGradientEquals(t, actp, expp)
// 		})

// 		t.Run("grad-tracked [4,4,4] base and [2,2,2] patch / Patch([1:3],[1:3],[1:3]) then BackPropagate / base gradient is 1 outside window 0 inside, patch gradient is all-ones", func(t *testing.T) {
// 			x, err := tensor.RandN([]int{4, 4, 4}, 0., 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			p, err := tensor.RandN([]int{2, 2, 2}, 0., 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := x.Patch([]tensor.Range{{From: 1, To: 3}, {From: 1, To: 3}, {From: 1, To: 3}}, p)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			actx := x.Gradient()
// 			actp := p.Gradient()

// 			expx, err := tensor.Of([][][]float64{
// 				{
// 					{1., 1., 1., 1.},
// 					{1., 1., 1., 1.},
// 					{1., 1., 1., 1.},
// 					{1., 1., 1., 1.},
// 				},
// 				{
// 					{1., 1., 1., 1.},
// 					{1., 0., 0., 1.},
// 					{1., 0., 0., 1.},
// 					{1., 1., 1., 1.},
// 				},
// 				{
// 					{1., 1., 1., 1.},
// 					{1., 0., 0., 1.},
// 					{1., 0., 0., 1.},
// 					{1., 1., 1., 1.},
// 				},
// 				{
// 					{1., 1., 1., 1.},
// 					{1., 1., 1., 1.},
// 					{1., 1., 1., 1.},
// 					{1., 1., 1., 1.},
// 				},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expp, err := tensor.Ones([]int{2, 2, 2}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, actx, expx)
// 			assertGradientEquals(t, actp, expp)
// 		})

// 		t.Run("grad-tracked [4,5] base, [2,2] patch p1 and p2 / sequential Patch then BackPropagate / gradients flow through both patch levels", func(t *testing.T) {
// 			x, err := tensor.RandN([]int{4, 5}, 0., 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			p1, err := tensor.RandN([]int{2, 2}, 0., 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			p2, err := tensor.RandN([]int{2, 2}, 0., 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			mid, err := x.Patch([]tensor.Range{{From: 0, To: 2}, {From: 0, To: 2}}, p1)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			y, err := mid.Patch([]tensor.Range{{From: 2, To: 4}, {From: 2, To: 4}}, p2)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			actx := x.Gradient()
// 			actp1 := p1.Gradient()
// 			actp2 := p2.Gradient()

// 			expx, err := tensor.Of([][]float64{
// 				{0., 0., 1., 1., 1.},
// 				{0., 0., 1., 1., 1.},
// 				{1., 1., 0., 0., 1.},
// 				{1., 1., 0., 0., 1.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expp1, err := tensor.Ones([]int{2, 2}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expp2, err := tensor.Ones([]int{2, 2}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, actx, expx)
// 			assertGradientEquals(t, actp1, expp1)
// 			assertGradientEquals(t, actp2, expp2)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("x untracked, p grad-tracked / Patch(nil) then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			x, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			p, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := x.Patch(nil, p)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("x grad-tracked, p untracked / Patch(nil) then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			x, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			p, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := x.Patch(nil, p)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("x untracked, p untracked / Patch(nil) then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			x, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			p, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := x.Patch(nil, p)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestScale(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, 2) grad-tracked / Scale(3) then BackPropagate / gradient of x is 3", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Scale(3.)
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 5) grad-tracked / Scale(-2) then BackPropagate / gradient of x is -2", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 5., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Scale(-2.)
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, -2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("RandN([3,4]) grad-tracked / Scale(-2) then BackPropagate / gradient of x is Full([3,4], -2)", func(t *testing.T) {
// 			x, err := tensor.RandN([]int{3, 4}, 0., 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Scale(-2.)
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full([]int{3, 4}, -2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 1) grad-tracked / Scale(2) then Scale(3) then BackPropagate / gradient of x is 6", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Scale(2.).Scale(3.)
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 6., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("x untracked / Scale(0) then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			x, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Scale(0.)
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestPow(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, 2) grad-tracked / Pow(3) then BackPropagate / gradient of x is 12", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Pow(3.)
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 12., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 5) grad-tracked / Pow(1) then BackPropagate / gradient of x is 1", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 5., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Pow(1.)
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 4) grad-tracked / Pow(0.5) then BackPropagate / gradient of x is 0.25", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 4., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Pow(0.5)
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 0.25, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 2) grad-tracked / Pow(2) then Pow(3) then BackPropagate / gradient of x is 192", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Pow(2.).Pow(3.)
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 192., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Of([1,3]{1,2,3}) grad-tracked / Pow(2) then BackPropagate / gradient of x is [2,4,6]", func(t *testing.T) {
// 			x, err := tensor.Of([][]float64{{1., 2., 3.}}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Pow(2.)
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Of([][]float64{{2., 4., 6.}}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("x untracked / Pow(0) then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			x, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Pow(0.)
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestExp(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, 1) grad-tracked / Exp then BackPropagate / gradient of x is e", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Exp()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, math.E, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 1) grad-tracked / Exp then Log then BackPropagate / gradient of x is 1", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Exp().Log()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("x untracked / Exp then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			x, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Exp()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestLog(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, 2) grad-tracked / Log then BackPropagate / gradient of x is 0.5", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Log()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 0.5, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 1) grad-tracked / Log then Exp then BackPropagate / gradient of x is 1", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Log().Exp()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("x untracked / Log then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			x, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Log()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestSin(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, π/3) grad-tracked / Sin then BackPropagate / gradient of x is cos(π/3)", func(t *testing.T) {
// 			x, err := tensor.Full(nil, math.Pi/3, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Sin()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, math.Cos(math.Pi/3), &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 0) grad-tracked / Sin then BackPropagate / gradient of x is cos(0) = 1", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 0., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Sin()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("x untracked / Sin then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			x, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Sin()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestCos(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, π/6) grad-tracked / Cos then BackPropagate / gradient of x is -sin(π/6)", func(t *testing.T) {
// 			x, err := tensor.Full(nil, math.Pi/6, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Cos()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, -math.Sin(math.Pi/6), &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 0) grad-tracked / Cos then BackPropagate / gradient of x is -sin(0) = 0", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 0., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Cos()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 0., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("x untracked / Cos then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			x, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Cos()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestTan(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, 0) grad-tracked / Tan then BackPropagate / gradient of x is sec²(0) = 1", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 0., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Tan()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, π/4) grad-tracked / Tan then BackPropagate / gradient of x is sec²(π/4)", func(t *testing.T) {
// 			x, err := tensor.Full(nil, math.Pi/4, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Tan()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, math.Pow(math.Cos(math.Pi/4), -2), &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("x untracked / Tan then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			x, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Tan()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestSinh(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, 0) grad-tracked / Sinh then BackPropagate / gradient of x is cosh(0) = 1", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 0., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Sinh()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 1) grad-tracked / Sinh then BackPropagate / gradient of x is cosh(1)", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Sinh()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, math.Cosh(1.), &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("x untracked / Sinh then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			x, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Sinh()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestCosh(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, 0) grad-tracked / Cosh then BackPropagate / gradient of x is sinh(0) = 0", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 0., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Cosh()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 0., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 1) grad-tracked / Cosh then BackPropagate / gradient of x is sinh(1)", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Cosh()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, math.Sinh(1.), &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("x untracked / Cosh then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			x, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Cosh()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestTanh(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, 0) grad-tracked / Tanh then BackPropagate / gradient of x is sech²(0) = 1", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 0., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Tanh()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 1) grad-tracked / Tanh then BackPropagate / gradient of x is sech²(1)", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Tanh()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, math.Pow(math.Cosh(1.), -2), &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("x untracked / Tanh then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			x, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y := x.Tanh()
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestElMax(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked / ElMax then BackPropagate / gradient of a is 1, gradient of b is 0", func(t *testing.T) {
// 			a, err := tensor.Full(nil, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.ElMax(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Full(nil, 0., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		t.Run("Full(nil, 2) and Full(nil, 2) both grad-tracked / ElMax then BackPropagate / gradient of a is 0.5, gradient of b is 0.5", func(t *testing.T) {
// 			a, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.ElMax(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Full(nil, 0.5, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Full(nil, 0.5, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		t.Run("same grad-tracked scalar tensor / a.ElMax(a) then BackPropagate / gradient of a is 1 (both tie contributions accumulate)", func(t *testing.T) {
// 			a, err := tensor.Full(nil, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.ElMax(a)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()

// 			expa, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 		})

// 		t.Run("grad-tracked [2,3] tensors with mixed a>b / a<b / a==b / ElMax then BackPropagate / per-element gradient pattern 0 or 1 or 0.5", func(t *testing.T) {
// 			a, err := tensor.Of([][]float64{
// 				{3., 1., 2.},
// 				{2., 3., 1.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Of([][]float64{
// 				{1., 3., 2.},
// 				{2., 1., 3.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.ElMax(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Of([][]float64{
// 				{1., 0., 0.5},
// 				{0.5, 1., 0.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Of([][]float64{
// 				{0., 1., 0.5},
// 				{0.5, 0., 1.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("a untracked, b grad-tracked / ElMax then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.ElMax(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a grad-tracked, b untracked / ElMax then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.ElMax(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a untracked, b untracked / ElMax then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.ElMax(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestElMin(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked / ElMin then BackPropagate / gradient of a is 0, gradient of b is 1", func(t *testing.T) {
// 			a, err := tensor.Full(nil, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.ElMin(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Full(nil, 0., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		t.Run("Full(nil, 2) and Full(nil, 2) both grad-tracked / ElMin then BackPropagate / gradient of a is 0.5, gradient of b is 0.5", func(t *testing.T) {
// 			a, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.ElMin(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Full(nil, 0.5, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Full(nil, 0.5, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		t.Run("same grad-tracked scalar tensor / a.ElMin(a) then BackPropagate / gradient of a is 1 (both tie contributions accumulate)", func(t *testing.T) {
// 			a, err := tensor.Full(nil, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.ElMin(a)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()

// 			expa, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 		})

// 		t.Run("grad-tracked [2,3] tensors with mixed a<b / a>b / a==b / ElMin then BackPropagate / per-element gradient pattern 0 or 1 or 0.5", func(t *testing.T) {
// 			a, err := tensor.Of([][]float64{
// 				{1., 3., 2.},
// 				{2., 1., 3.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Of([][]float64{
// 				{3., 1., 2.},
// 				{2., 3., 1.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.ElMin(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Of([][]float64{
// 				{1., 0., 0.5},
// 				{0.5, 1., 0.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Of([][]float64{
// 				{0., 1., 0.5},
// 				{0.5, 0., 1.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("a untracked, b grad-tracked / ElMin then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.ElMin(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a grad-tracked, b untracked / ElMin then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.ElMin(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a untracked, b untracked / ElMin then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.ElMin(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestAdd(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked / Add then BackPropagate / gradient of a is 1, gradient of b is 1", func(t *testing.T) {
// 			a, err := tensor.Full(nil, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Add(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		t.Run("same grad-tracked scalar tensor / x.Add(x) then BackPropagate / gradient of x is 2", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := x.Add(x)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("same grad-tracked scalar tensor / x.Add(x).Add(x) then BackPropagate / gradient of x is 3", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			tmp, err := x.Add(x)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			y, err := tmp.Add(x)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("two grad-tracked [2,3] tensors / Add then BackPropagate / gradient of each is all-ones [2,3]", func(t *testing.T) {
// 			a, err := tensor.RandN([]int{2, 3}, 0., 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.RandN([]int{2, 3}, 0., 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Add(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Ones([]int{2, 3}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Ones([]int{2, 3}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("a untracked, b grad-tracked / Add then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Add(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a grad-tracked, b untracked / Add then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Add(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a untracked, b untracked / Add then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Add(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestSub(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked / Sub then BackPropagate / gradient of a is 1, gradient of b is -1", func(t *testing.T) {
// 			a, err := tensor.Full(nil, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Sub(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Full(nil, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Full(nil, -1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		t.Run("same grad-tracked scalar tensor / x.Sub(x) then BackPropagate / gradient of x is 0", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 5., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := x.Sub(x)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 0., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 3) grad-tracked / x.Sub(x.Scale(2)) then BackPropagate / gradient of x is -1", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := x.Sub(x.Scale(2.))
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, -1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("two grad-tracked [2,3] tensors / Sub then BackPropagate / gradient of a is all-ones [2,3], gradient of b is all-neg-ones [2,3]", func(t *testing.T) {
// 			a, err := tensor.RandN([]int{2, 3}, 0., 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.RandN([]int{2, 3}, 0., 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Sub(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Full([]int{2, 3}, -1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("a untracked, b grad-tracked / Sub then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Sub(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a grad-tracked, b untracked / Sub then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Sub(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a untracked, b untracked / Sub then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Sub(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestMul(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked / Mul then BackPropagate / gradient of a is 2, gradient of b is 3", func(t *testing.T) {
// 			a, err := tensor.Full(nil, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Mul(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Full(nil, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		t.Run("Of([2,2]{1,2,3,4}) grad-tracked / x.Mul(x) then BackPropagate / gradient of x is 2x element-wise", func(t *testing.T) {
// 			x, err := tensor.Of([][]float64{
// 				{1., 2.},
// 				{3., 4.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := x.Mul(x)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Of([][]float64{
// 				{2., 4.},
// 				{6., 8.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 2) grad-tracked / x.Mul(x).Mul(x) then BackPropagate / gradient of x is 3x²=12", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			tmp, err := x.Mul(x)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			y, err := tmp.Mul(x)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, 12., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("a untracked, b grad-tracked / Mul then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Mul(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a grad-tracked, b untracked / Mul then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Mul(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a untracked, b untracked / Mul then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Mul(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestDiv(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked / Div then BackPropagate / gradient of a is 0.5, gradient of b is -0.75", func(t *testing.T) {
// 			a, err := tensor.Full(nil, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Div(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Full(nil, 0.5, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Full(nil, -0.75, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		t.Run("Of([2,2]{1,2,3,4}) grad-tracked / x.Div(x) then BackPropagate / gradient of x is all-zeros", func(t *testing.T) {
// 			x, err := tensor.Of([][]float64{
// 				{1., 2.},
// 				{3., 4.},
// 			}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := x.Div(x)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Zeros([]int{2, 2}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full(nil, 2) grad-tracked / x.Div(x.Pow(2)) then BackPropagate / gradient of x is -1/x²=-0.25", func(t *testing.T) {
// 			x, err := tensor.Full(nil, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := x.Div(x.Pow(2.))
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full(nil, -0.25, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("a untracked, b grad-tracked / Div then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Div(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a grad-tracked, b untracked / Div then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Div(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a untracked, b untracked / Div then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros(nil, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Div(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestDot(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full([2], 2) and Full([2], 3) both grad-tracked / Dot then BackPropagate / gradient of a is Full([2], 3), gradient of b is Full([2], 2)", func(t *testing.T) {
// 			a, err := tensor.Full([]int{2}, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Full([]int{2}, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Dot(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Full([]int{2}, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Full([]int{2}, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		t.Run("Of([3]{1,2,3}) grad-tracked / x.Dot(x) then BackPropagate / gradient of x is 2x=[2,4,6]", func(t *testing.T) {
// 			x, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := x.Dot(x)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Of([]float64{2., 4., 6.}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Of([4]{1,2,3,4}) and Of([4]{4,3,2,1}) both grad-tracked / Dot then BackPropagate / gradient of a is b, gradient of b is a", func(t *testing.T) {
// 			a, err := tensor.Of([]float64{1., 2., 3., 4.}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Of([]float64{4., 3., 2., 1.}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Dot(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Of([]float64{4., 3., 2., 1.}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Of([]float64{1., 2., 3., 4.}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("a untracked, b grad-tracked / Dot then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros([]int{1}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros([]int{1}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Dot(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a grad-tracked, b untracked / Dot then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros([]int{1}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros([]int{1}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Dot(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a untracked, b untracked / Dot then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			a, err := tensor.Zeros([]int{1}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros([]int{1}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.Dot(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

// func TestMatmul(t *testing.T) {
// 	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

// 		// ============================== main paths ==============================

// 		t.Run("Full([2,3], 2) and Full([3,2], 3) both grad-tracked / MatMul then BackPropagate / gradient of a is Full([2,3], 6), gradient of b is Full([3,2], 4)", func(t *testing.T) {
// 			a, err := tensor.Full([]int{2, 3}, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Full([]int{3, 2}, 3., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.MatMul(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			acta := a.Gradient()
// 			actb := b.Gradient()

// 			expa, err := tensor.Full([]int{2, 3}, 6., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			expb, err := tensor.Full([]int{3, 2}, 4., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, acta, expa)
// 			assertGradientEquals(t, actb, expb)
// 		})

// 		t.Run("Eye(3) grad-tracked / x.MatMul(x) then BackPropagate / gradient of x is Full([3,3], 2)", func(t *testing.T) {
// 			x, err := tensor.Eye(3, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := x.MatMul(x)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full([]int{3, 3}, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		t.Run("Full([2,3],1) grad-tracked, Full([3,2],1) and Zeros([2,2]) untracked / x.MatMul(W).Add(b) then BackPropagate / gradient of x is Full([2,3], 2)", func(t *testing.T) {
// 			x, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			w, err := tensor.Full([]int{3, 2}, 1., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Zeros([]int{2, 2}, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			mm, err := x.MatMul(w)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			y, err := mm.Add(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			act := x.Gradient()

// 			exp, err := tensor.Full([]int{2, 3}, 2., &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			assertGradientEquals(t, act, exp)
// 		})

// 		// ============================== untracked paths ==============================

// 		t.Run("a untracked, b grad-tracked / MatMul then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Eye(1, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Eye(1, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.MatMul(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a grad-tracked, b untracked / MatMul then BackPropagate / y has non-nil gradient", func(t *testing.T) {
// 			a, err := tensor.Eye(1, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: true,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Eye(1, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.MatMul(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() == nil {
// 				t.Fatal("expected gradient not to be nil")
// 			}
// 		})

// 		t.Run("a untracked, b untracked / MatMul then BackPropagate / y has nil gradient", func(t *testing.T) {
// 			a, err := tensor.Eye(1, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			b, err := tensor.Eye(1, &tensor.Config{
// 				Device:    dev,
// 				GradTrack: false,
// 			})
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			y, err := a.MatMul(b)
// 			if err != nil {
// 				t.Fatal(err)
// 			}
// 			err = tensor.BackPropagate(y)
// 			if err != nil {
// 				t.Fatal(err)
// 			}

// 			if y.Gradient() != nil {
// 				t.Fatal("expected gradient to be nil")
// 			}
// 		})
// 	})
// }

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
