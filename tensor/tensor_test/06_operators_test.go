package tensor_test

import (
	"fmt"
	"math"
	"sync"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestScale(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor with value 1 | Scale(0) | returns scalar 0", func(t *testing.T) {
			x, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(0.)

			h, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [-1, 0, 1] | Scale(-1) | returns [1, 0, -1]", func(t *testing.T) {
			x, err := tensor.Of([]float64{-1., 0., 1.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(-1.)

			h, err := tensor.Of([]float64{1., 0., -1.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor with mixed values | Scale(0.5) | returns halved values", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{-5., 2.},
				{3., -4.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(0.5)

			h, err := tensor.Of([][]float64{
				{-2.5, 1.},
				{1.5, -2.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros tensor [1,2,3,4] | Scale(5) | returns zeros tensor", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(5.)

			h, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Scale(2.) then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(2.)

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | Scale(2.) | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(2.)

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | Scale(2.) | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(2.)

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | Scale(2.) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(2.)

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Scale(2.) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(2.)

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Scale(2.) then BackPropagate | gradient is nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(2.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("Full(nil, 2) grad-tracked | Scale(3) then BackPropagate | gradient of x is 3", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Full(nil, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full(nil, 5) grad-tracked | Scale(-2) then BackPropagate | gradient of x is -2", func(t *testing.T) {
			x, err := tensor.Full(nil, 5., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(-2.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full(nil, -2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,4], 1) grad-tracked | Scale(-2) then BackPropagate | gradient of x is Full([3,4], -2)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(-2.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full([]int{3, 4}, -2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 1 | Scale(2.) | returns Full([1,2^20], 2)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(2.)

			h, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor filled with 1 | Scale(2.) then BackPropagate | gradient of x is Full([1,2^20], 2)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(2.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor filled with 1 | concurrent repeated Scale(2.) over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y := x.Scale(2.)

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] tensor | Scale(2.) then ResetGradient(source, false) | result is still gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Scale(2.)

			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})
	})
}

func TestPow(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor with value e | Pow(0) | returns scalar 1", func(t *testing.T) {
			x, err := tensor.Of(math.E, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(0.)

			h, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [-1, 0, 1] | Pow(1) | returns unchanged tensor", func(t *testing.T) {
			x, err := tensor.Of([]float64{-1., 0., 1.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(1.)

			h, err := tensor.Of([]float64{-1., 0., 1.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor with mixed values | Pow(-2) | returns reciprocal squares", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{-2., 2.},
				{-1., 0.5},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(-2)

			h, err := tensor.Of([][]float64{
				{0.25, 0.25},
				{1., 4.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("ones tensor [1,2,3,4] | Pow(1000) | returns ones tensor", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(1000)

			h, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Pow(2.) then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(2.)

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | Pow(2.) | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(2.)

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | Pow(2.) | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(2.)

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | Pow(2.) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(2.)

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Pow(2.) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(2.)

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Pow(2.) then BackPropagate | gradient is nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(2.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("Full(nil, 2) grad-tracked | Pow(3) then BackPropagate | gradient of x is 12", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Full(nil, 12., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full(nil, 5) grad-tracked | Pow(1) then BackPropagate | gradient of x is 1", func(t *testing.T) {
			x, err := tensor.Full(nil, 5., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(1.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full(nil, 4) grad-tracked | Pow(0.5) then BackPropagate | gradient of x is 0.25", func(t *testing.T) {
			x, err := tensor.Full(nil, 4., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(0.5)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full(nil, 0.25, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([1,3]{1,2,3}) grad-tracked | Pow(2) then BackPropagate | gradient of x is [2,4,6]", func(t *testing.T) {
			x, err := tensor.Of([][]float64{{1., 2., 3.}}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(2.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Of([][]float64{{2., 4., 6.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 3 | Pow(2.) | returns Full([1,2^20], 9)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(2.)

			h, err := tensor.Full([]int{1, n}, 9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor filled with 3 | Pow(2.) then BackPropagate | gradient of x is Full([1,2^20], 6)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(2.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full([]int{1, n}, 6., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor filled with 3 | concurrent repeated Pow(2.) over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y := x.Pow(2.)

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] tensor | Pow(2.) then ResetGradient(source, false) | result is still gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Pow(2.)

			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})
	})
}

func TestExp(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor with value 1 | Exp() | returns e", func(t *testing.T) {
			x, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Exp()

			h, err := tensor.Of(math.E, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [-1, 0, 1] | Exp() | returns [1/e, 1, e]", func(t *testing.T) {
			x, err := tensor.Of([]float64{-1., 0., 1.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Exp()

			h, err := tensor.Of([]float64{1 / math.E, 1., math.E}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor with positive and negative values | Exp() | returns e-powers", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{1., -1.},
				{-2., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Exp()

			h, err := tensor.Of([][]float64{
				{math.E, 1 / math.E},
				{1 / (math.E * math.E), 1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros tensor [1,2,3,4] | Exp() | returns ones tensor", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Exp()

			h, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Exp() then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Exp()

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | Exp() | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Exp()

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | Exp() | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Exp()

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | Exp() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Exp()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Exp() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Exp()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Exp() then BackPropagate | gradient is nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("Full(nil, 1) grad-tracked | Exp then BackPropagate | gradient of x is e", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Full(nil, math.E, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 1 | Exp() | returns Full([1,2^20], e)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Exp()

			h, err := tensor.Full([]int{1, n}, math.E, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor filled with 1 | Exp() then BackPropagate | gradient of x is Full([1,2^20], e)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{1, n}, math.E, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor filled with 1 | concurrent repeated Exp() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, math.E, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y := x.Exp()

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] tensor | Exp() then ResetGradient(source, false) | result is still gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Exp()

			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})
	})
}

func TestLog(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor with value -1 | Log() | returns NaN", func(t *testing.T) {
			x, err := tensor.Of(-1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()

			if val, err := y.At(); err != nil {
				t.Fatal(err)
			} else if !math.IsNaN(val) {
				t.Fatalf("expected scalar tensor value to be (NaN): got (%f)", val)
			}
		})

		t.Run("scalar tensor with value 0 | Log() | returns -Inf", func(t *testing.T) {
			x, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()

			if val, err := y.At(); err != nil {
				t.Fatal(err)
			} else if !math.IsInf(val, -1) {
				t.Fatalf("expected scalar tensor value to be (-Inf): got (%f)", val)
			}
		})

		t.Run("scalar tensor with value e | Log() | returns 1", func(t *testing.T) {
			x, err := tensor.Of(math.E, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()

			h, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, e, e^2] | Log() | returns [0, 1, 2]", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., math.E, math.E * math.E}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()

			h, err := tensor.Of([]float64{0., 1., 2.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor [[e, 1]] | Log() | returns [[1, 0]]", func(t *testing.T) {
			x, err := tensor.Of([][]float64{{math.E, 1.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()

			h, err := tensor.Of([][]float64{{1., 0.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("ones tensor [1,2,3,4] | Log() | returns zeros tensor", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()

			h, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Log() then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | Log() | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | Log() | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | Log() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Log() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Log() then BackPropagate | gradient is nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("Full(nil, 2) grad-tracked | Log then BackPropagate | gradient of x is 0.5", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Full(nil, 0.5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 1 | Log() | returns Full([1,2^20], 0)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()

			h, err := tensor.Full([]int{1, n}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor filled with 1 | Log() then BackPropagate | gradient of x is Full([1,2^20], 1)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor filled with 1 | concurrent repeated Log() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y := x.Log()

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] tensor | Log() then ResetGradient(source, false) | result is still gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Log()

			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})
	})
}

func TestSin(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor with value 0 | Sin() | returns 0", func(t *testing.T) {
			x, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sin()

			h, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("scalar tensor with value pi/6 | Sin() | returns 0.5", func(t *testing.T) {
			x, err := tensor.Of(math.Pi/6, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sin()

			h, err := tensor.Of(0.5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("scalar tensor with value pi/2 | Sin() | returns 1", func(t *testing.T) {
			x, err := tensor.Of(math.Pi/2, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sin()

			h, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros tensor [1,2,3,4] | Sin() | returns zeros tensor", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sin()

			h, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Sin() then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sin()

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | Sin() | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sin()

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | Sin() | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sin()

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | Sin() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sin()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Sin() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sin()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Sin() then BackPropagate | gradient is nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("Full(nil, π/3) grad-tracked | Sin then BackPropagate | gradient of x is cos(π/3)", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Full(nil, math.Cos(math.Pi/3), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full(nil, 0) grad-tracked | Sin then BackPropagate | gradient of x is cos(0) = 1", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with π/2 | Sin() | returns Full([1,2^20], 1)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, math.Pi/2, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sin()

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor filled with π/2 | Sin() then BackPropagate | gradient of x is Full([1,2^20], 0)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, math.Pi/2, &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{1, n}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor filled with π/2 | concurrent repeated Sin() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, math.Pi/2, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y := x.Sin()

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] tensor | Sin() then ResetGradient(source, false) | result is still gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sin()

			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})
	})
}

func TestCos(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor with value 0 | Cos() | returns 1", func(t *testing.T) {
			x, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cos()

			h, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("scalar tensor with value pi/3 | Cos() | returns 0.5", func(t *testing.T) {
			x, err := tensor.Of(math.Pi/3, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cos()

			h, err := tensor.Of(0.5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("scalar tensor with value pi/2 | Cos() | returns 0", func(t *testing.T) {
			x, err := tensor.Of(math.Pi/2, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cos()

			h, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros tensor [1,2,3,4] | Cos() | returns ones tensor", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cos()

			h, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Cos() then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cos()

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | Cos() | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cos()

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | Cos() | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cos()

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | Cos() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cos()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Cos() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cos()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Cos() then BackPropagate | gradient is nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("Full(nil, π/6) grad-tracked | Cos then BackPropagate | gradient of x is -sin(π/6)", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Full(nil, -math.Sin(math.Pi/6), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full(nil, 0) grad-tracked | Cos then BackPropagate | gradient of x is -sin(0) = 0", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 1 | Cos() | returns Full([1,2^20], cos(1))", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cos()

			h, err := tensor.Full([]int{1, n}, math.Cos(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor filled with 1 | Cos() then BackPropagate | gradient of x is Full([1,2^20], -sin(1))", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{1, n}, -math.Sin(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor filled with 1 | concurrent repeated Cos() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, math.Cos(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y := x.Cos()

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] tensor | Cos() then ResetGradient(source, false) | result is still gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cos()

			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})
	})
}

func TestTan(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor with value 0 | Tan() | returns 0", func(t *testing.T) {
			x, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tan()

			h, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("scalar tensor with value pi/4 | Tan() | returns 1", func(t *testing.T) {
			x, err := tensor.Of(math.Pi/4, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tan()

			h, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros tensor [1,2,3,4] | Tan() | returns zeros tensor", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tan()

			h, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Tan() then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tan()

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | Tan() | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tan()

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | Tan() | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tan()

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | Tan() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tan()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Tan() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tan()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Tan() then BackPropagate | gradient is nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("Full(nil, 0) grad-tracked | Tan then BackPropagate | gradient of x is sec²(0) = 1", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full(nil, π/4) grad-tracked | Tan then BackPropagate | gradient of x is sec²(π/4)", func(t *testing.T) {
			x, err := tensor.Full(nil, math.Pi/4, &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full(nil, math.Pow(math.Cos(math.Pi/4), -2), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 1 | Tan() | returns Full([1,2^20], tan(1))", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tan()

			h, err := tensor.Full([]int{1, n}, math.Tan(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor filled with 1 | Tan() then BackPropagate | gradient of x is Full([1,2^20], sec²(1))", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{1, n}, math.Pow(math.Cos(1.), -2), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor filled with 1 | concurrent repeated Tan() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, math.Tan(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y := x.Tan()

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] tensor | Tan() then ResetGradient(source, false) | result is still gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tan()

			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})
	})
}

func TestSinh(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor with value 0 | Sinh() | returns 0", func(t *testing.T) {
			x, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sinh()

			h, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("scalar tensor with value 1 | Sinh() | returns (e^2-1)/(2e)", func(t *testing.T) {
			x, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sinh()

			h, err := tensor.Of((math.E*math.E-1)/(2*math.E), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros tensor [1,2,3,4] | Sinh() | returns zeros tensor", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sinh()

			h, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Sinh() then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sinh()

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | Sinh() | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sinh()

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | Sinh() | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sinh()

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | Sinh() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sinh()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Sinh() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sinh()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Sinh() then BackPropagate | gradient is nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("Full(nil, 0) grad-tracked | Sinh then BackPropagate | gradient of x is cosh(0) = 1", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full(nil, 1) grad-tracked | Sinh then BackPropagate | gradient of x is cosh(1)", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Full(nil, math.Cosh(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 1 | Sinh() | returns Full([1,2^20], sinh(1))", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sinh()

			h, err := tensor.Full([]int{1, n}, math.Sinh(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor filled with 1 | Sinh() then BackPropagate | gradient of x is Full([1,2^20], cosh(1))", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{1, n}, math.Cosh(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor filled with 1 | concurrent repeated Sinh() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, math.Sinh(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y := x.Sinh()

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] tensor | Sinh() then ResetGradient(source, false) | result is still gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Sinh()

			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})
	})
}

func TestCosh(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor with value 0 | Cosh() | returns 1", func(t *testing.T) {
			x, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cosh()

			h, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("scalar tensor with value 1 | Cosh() | returns (e^2+1)/(2e)", func(t *testing.T) {
			x, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cosh()

			h, err := tensor.Of((math.E*math.E+1)/(2*math.E), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros tensor [1,2,3,4] | Cosh() | returns ones tensor", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cosh()

			h, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Cosh() then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cosh()

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | Cosh() | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cosh()

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | Cosh() | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cosh()

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | Cosh() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cosh()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Cosh() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cosh()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Cosh() then BackPropagate | gradient is nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("Full(nil, 0) grad-tracked | Cosh then BackPropagate | gradient of x is sinh(0) = 0", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full(nil, 1) grad-tracked | Cosh then BackPropagate | gradient of x is sinh(1)", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Full(nil, math.Sinh(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 1 | Cosh() | returns Full([1,2^20], cosh(1))", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cosh()

			h, err := tensor.Full([]int{1, n}, math.Cosh(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor filled with 1 | Cosh() then BackPropagate | gradient of x is Full([1,2^20], sinh(1))", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{1, n}, math.Sinh(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor filled with 1 | concurrent repeated Cosh() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, math.Cosh(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y := x.Cosh()

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] tensor | Cosh() then ResetGradient(source, false) | result is still gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Cosh()

			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})
	})
}

func TestTanh(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor with value 0 | Tanh() | returns 0", func(t *testing.T) {
			x, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tanh()

			h, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("scalar tensor with value -Inf | Tanh() | returns -1", func(t *testing.T) {
			x, err := tensor.Of(math.Inf(-1), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tanh()

			h, err := tensor.Of(-1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("scalar tensor with value +Inf | Tanh() | returns 1", func(t *testing.T) {
			x, err := tensor.Of(math.Inf(+1), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tanh()

			h, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("scalar tensor with value 1 | Tanh() | returns (e^2-1)/(e^2+1)", func(t *testing.T) {
			x, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tanh()

			h, err := tensor.Of((math.E*math.E-1)/(math.E*math.E+1), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros tensor [1,2,3,4] | Tanh() | returns zeros tensor", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tanh()

			h, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Tanh() then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tanh()

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | Tanh() | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tanh()

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | Tanh() | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tanh()

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | Tanh() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tanh()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Tanh() | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tanh()

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Tanh() then BackPropagate | gradient is nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("Full(nil, 0) grad-tracked | Tanh then BackPropagate | gradient of x is sech²(0) = 1", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full(nil, 1) grad-tracked | Tanh then BackPropagate | gradient of x is sech²(1)", func(t *testing.T) {
			x, err := tensor.Full(nil, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full(nil, math.Pow(math.Cosh(1.), -2), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 1 | Tanh() | returns Full([1,2^20], tanh(1))", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tanh()

			h, err := tensor.Full([]int{1, n}, math.Tanh(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor filled with 1 | Tanh() then BackPropagate | gradient of x is Full([1,2^20], sech²(1))", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{1, n}, math.Pow(math.Cosh(1.), -2), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor filled with 1 | concurrent repeated Tanh() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, math.Tanh(1.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y := x.Tanh()

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] tensor | Tanh() then ResetGradient(source, false) | result is still gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := x.Tanh()

			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})
	})
}

func TestEq(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar 1 | Eq(scalar 1) | returns scalar 1", func(t *testing.T) {
			x1, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Eq(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [0, e+eps] | Eq([0, e]) | returns [1, 0]", func(t *testing.T) {
			x1, err := tensor.Of([]float64{0., math.E + 1e-15}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{0., math.E}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Eq(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{1., 0.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor [[1,2,3],[-1,-2,-3]] | Eq([[-1,-2,-3],[1,2,3]]) | returns all zeros", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{-1., -2., -3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{-1., -2., -3.},
				{1., 2., 3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Eq(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{0., 0., 0.},
				{0., 0., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("ones tensor [1,2,3,4] | Eq(ones tensor [1,2,3,4]) | returns ones tensor", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Eq(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | Eq(itself) | returns all ones", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Eq(x)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Eq([3,4] tensor) then Device() | returns the device the inputs were created on", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Eq(x2)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("two untracked [3,4] tensors | Eq | y is not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Eq(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor and untracked [3,4] tensor | Eq | y is not gradient-tracked because operation is not differentiable", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Eq(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("untracked [3,4] tensor and grad-tracked [3,4] tensor | Eq | y is not gradient-tracked because operation is not differentiable", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Eq(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("two grad-tracked [3,4] tensors | Eq | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Eq(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked [3,4] tensors | Eq | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Eq(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked [3,4] tensors | Eq then BackPropagate | gradient is nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Eq(x2)
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

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensors both filled with 7 | Eq | returns Full([1,2^20], 1)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Eq(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensors both filled with 7 | concurrent repeated Eq over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x1, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x1.Eq(x2)
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] x1 and untracked [3,4] x2 | Eq then ResetGradient(x1, false) | result stays not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Eq(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x1, false)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("untracked [3,4] x1 and grad-tracked [3,4] x2 | Eq then ResetGradient(x2, false) | result stays not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Eq(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x2, false)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("[6,5,2] tensor | Eq(nil) | returns error: device mismatch", func(t *testing.T) {
			x, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Eq(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Eq tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor and tensor [1] | Eq | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Eq(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Eq tensors' dimension validation failed: expected number of dimensions to match: (0) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1] and [1,1] | Eq | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Eq(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Eq tensors' dimension validation failed: expected number of dimensions to match: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,5] | Eq | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Eq(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Eq tensors' dimension validation failed: expected number of dimensions to match: (3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1,2] and [2,1] | Eq | returns error: size mismatch at dimension 0", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Eq(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Eq tensors' dimension validation failed: expected sizes to match at dimension (0): (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,4,2] | Eq | returns error: size mismatch at dimension 1", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 4, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Eq(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Eq tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2,2], 1.) on d1 and Full([2,2], 1.) on d2 | Eq | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Eq(x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("Eq tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestNe(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar 1 | Ne(scalar 1) | returns scalar 0", func(t *testing.T) {
			x1, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ne(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [0, e+eps] | Ne([0, e]) | returns [0, 1]", func(t *testing.T) {
			x1, err := tensor.Of([]float64{0., math.E + 1e-15}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{0., math.E}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ne(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{0., 1.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor [[1,2,3],[-1,-2,-3]] | Ne([[-1,-2,-3],[1,2,3]]) | returns all ones", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{-1., -2., -3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{-1., -2., -3.},
				{1., 2., 3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ne(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{1., 1., 1.},
				{1., 1., 1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("ones tensor [1,2,3,4] | Ne(ones tensor [1,2,3,4]) | returns zeros tensor", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ne(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | Ne(itself) | returns all zeros", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Ne(x)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Ne([3,4] tensor) then Device() | returns the device the inputs were created on", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ne(x2)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("two untracked [3,4] tensors | Ne | y is not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ne(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor and untracked [3,4] tensor | Ne | y is not gradient-tracked because operation is not differentiable", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ne(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("untracked [3,4] tensor and grad-tracked [3,4] tensor | Ne | y is not gradient-tracked because operation is not differentiable", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ne(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("two grad-tracked [3,4] tensors | Ne | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ne(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked [3,4] tensors | Ne | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ne(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked [3,4] tensors | Ne then BackPropagate | gradient is nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ne(x2)
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

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensors filled with 7 and 8 | Ne | returns Full([1,2^20], 1)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 8., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ne(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensors filled with 7 and 8 | concurrent repeated Ne over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x1, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 8., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x1.Ne(x2)
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] x1 and untracked [3,4] x2 | Ne then ResetGradient(x1, false) | result stays not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ne(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x1, false)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("untracked [3,4] x1 and grad-tracked [3,4] x2 | Ne then ResetGradient(x2, false) | result stays not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ne(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x2, false)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("[6,5,2] tensor | Ne(nil) | returns error: device mismatch", func(t *testing.T) {
			x, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Ne(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Ne tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor and tensor [1] | Ne | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Ne(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Ne tensors' dimension validation failed: expected number of dimensions to match: (0) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1] and [1,1] | Ne | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Ne(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Ne tensors' dimension validation failed: expected number of dimensions to match: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,5] | Ne | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Ne(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Ne tensors' dimension validation failed: expected number of dimensions to match: (3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1,2] and [2,1] | Ne | returns error: size mismatch at dimension 0", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Ne(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Ne tensors' dimension validation failed: expected sizes to match at dimension (0): (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,4,2] | Ne | returns error: size mismatch at dimension 1", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 4, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Ne(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Ne tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2,2], 1.) on d1 and Full([2,2], 1.) on d2 | Ne | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Ne(x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("Ne tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestGt(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar 1 | Gt(scalar 1) | returns scalar 0", func(t *testing.T) {
			x1, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Gt(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [0, e+eps] | Gt([0, e]) | returns [0, 1]", func(t *testing.T) {
			x1, err := tensor.Of([]float64{0., math.E + 1e-15}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{0., math.E}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Gt(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{0., 1.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor [[1,2,3],[-1,-2,-3]] | Gt([[-1,-2,-3],[1,2,3]]) | returns [[1,1,1],[0,0,0]]", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{-1., -2., -3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{-1., -2., -3.},
				{1., 2., 3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Gt(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{1., 1., 1.},
				{0., 0., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros tensor [1,2,3,4] | Gt(ones tensor [1,2,3,4]) | returns zeros tensor", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Gt(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | Gt(itself) | returns all zeros", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Gt(x)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Gt([3,4] tensor) then Device() | returns the device the inputs were created on", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Gt(x2)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("two untracked [3,4] tensors | Gt | y is not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Gt(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor and untracked [3,4] tensor | Gt | y is not gradient-tracked because operation is not differentiable", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Gt(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("untracked [3,4] tensor and grad-tracked [3,4] tensor | Gt | y is not gradient-tracked because operation is not differentiable", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Gt(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("two grad-tracked [3,4] tensors | Gt | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Gt(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked [3,4] tensors | Gt | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Gt(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked [3,4] tensors | Gt then BackPropagate | gradient is nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Gt(x2)
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

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensors filled with 8 and 7 | Gt | returns Full([1,2^20], 1)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 8., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Gt(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensors filled with 8 and 7 | concurrent repeated Gt over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x1, err := tensor.Full([]int{1, n}, 8., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x1.Gt(x2)
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] x1 and untracked [3,4] x2 | Gt then ResetGradient(x1, false) | result stays not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Gt(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x1, false)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("untracked [3,4] x1 and grad-tracked [3,4] x2 | Gt then ResetGradient(x2, false) | result stays not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Gt(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x2, false)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("[6,5,2] tensor | Gt(nil) | returns error: device mismatch", func(t *testing.T) {
			x, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Gt(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Gt tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor and tensor [1] | Gt | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Gt(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Gt tensors' dimension validation failed: expected number of dimensions to match: (0) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1] and [1,1] | Gt | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Gt(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Gt tensors' dimension validation failed: expected number of dimensions to match: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,5] | Gt | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Gt(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Gt tensors' dimension validation failed: expected number of dimensions to match: (3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1,2] and [2,1] | Gt | returns error: size mismatch at dimension 0", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Gt(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Gt tensors' dimension validation failed: expected sizes to match at dimension (0): (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,4,2] | Gt | returns error: size mismatch at dimension 1", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 4, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Gt(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Gt tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2,2], 1.) on d1 and Full([2,2], 1.) on d2 | Gt | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Gt(x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("Gt tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestGe(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar 1 | Ge(scalar 1) | returns scalar 1", func(t *testing.T) {
			x1, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ge(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [0, e+eps] | Ge([0, e]) | returns [1, 1]", func(t *testing.T) {
			x1, err := tensor.Of([]float64{0., math.E + 1e-15}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{0., math.E}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ge(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{1., 1.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor [[1,2,3],[-1,-2,-3]] | Ge([[-1,-2,-3],[1,2,3]]) | returns [[1,1,1],[0,0,0]]", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{-1., -2., -3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{-1., -2., -3.},
				{1., 2., 3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ge(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{1., 1., 1.},
				{0., 0., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros tensor [1,2,3,4] | Ge(ones tensor [1,2,3,4]) | returns zeros tensor", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ge(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | Ge(itself) | returns all ones", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Ge(x)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Ge([3,4] tensor) then Device() | returns the device the inputs were created on", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ge(x2)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("two untracked [3,4] tensors | Ge | y is not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ge(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor and untracked [3,4] tensor | Ge | y is not gradient-tracked because operation is not differentiable", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ge(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("untracked [3,4] tensor and grad-tracked [3,4] tensor | Ge | y is not gradient-tracked because operation is not differentiable", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ge(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("two grad-tracked [3,4] tensors | Ge | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ge(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked [3,4] tensors | Ge | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ge(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked [3,4] tensors | Ge then BackPropagate | gradient is nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ge(x2)
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

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensors both filled with 7 | Ge | returns Full([1,2^20], 1)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ge(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensors both filled with 7 | concurrent repeated Ge over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x1, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x1.Ge(x2)
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] x1 and untracked [3,4] x2 | Ge then ResetGradient(x1, false) | result stays not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ge(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x1, false)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("untracked [3,4] x1 and grad-tracked [3,4] x2 | Ge then ResetGradient(x2, false) | result stays not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Ge(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x2, false)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("[6,5,2] tensor | Ge(nil) | returns error: device mismatch", func(t *testing.T) {
			x, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Ge(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Ge tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor and tensor [1] | Ge | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Ge(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Ge tensors' dimension validation failed: expected number of dimensions to match: (0) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1] and [1,1] | Ge | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Ge(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Ge tensors' dimension validation failed: expected number of dimensions to match: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,5] | Ge | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Ge(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Ge tensors' dimension validation failed: expected number of dimensions to match: (3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1,2] and [2,1] | Ge | returns error: size mismatch at dimension 0", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Ge(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Ge tensors' dimension validation failed: expected sizes to match at dimension (0): (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,4,2] | Ge | returns error: size mismatch at dimension 1", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 4, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Ge(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Ge tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2,2], 1.) on d1 and Full([2,2], 1.) on d2 | Ge | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Ge(x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("Ge tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestLt(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar 1 | Lt(scalar 1) | returns scalar 0", func(t *testing.T) {
			x1, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Lt(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [0, e+eps] | Lt([0, e]) | returns [0, 0]", func(t *testing.T) {
			x1, err := tensor.Of([]float64{0., math.E + 1e-15}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{0., math.E}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Lt(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{0., 0.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor [[1,2,3],[-1,-2,-3]] | Lt([[-1,-2,-3],[1,2,3]]) | returns [[0,0,0],[1,1,1]]", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{-1., -2., -3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{-1., -2., -3.},
				{1., 2., 3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Lt(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{0., 0., 0.},
				{1., 1., 1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros tensor [1,2,3,4] | Lt(ones tensor [1,2,3,4]) | returns ones tensor", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Lt(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | Lt(itself) | returns all zeros", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Lt(x)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Lt([3,4] tensor) then Device() | returns the device the inputs were created on", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Lt(x2)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("two untracked [3,4] tensors | Lt | y is not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Lt(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor and untracked [3,4] tensor | Lt | y is not gradient-tracked because operation is not differentiable", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Lt(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("untracked [3,4] tensor and grad-tracked [3,4] tensor | Lt | y is not gradient-tracked because operation is not differentiable", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Lt(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("two grad-tracked [3,4] tensors | Lt | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Lt(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked [3,4] tensors | Lt | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Lt(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked [3,4] tensors | Lt then BackPropagate | gradient is nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Lt(x2)
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

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensors filled with 7 and 8 | Lt | returns Full([1,2^20], 1)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 8., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Lt(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensors filled with 7 and 8 | concurrent repeated Lt over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x1, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 8., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x1.Lt(x2)
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] x1 and untracked [3,4] x2 | Lt then ResetGradient(x1, false) | result stays not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Lt(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x1, false)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("untracked [3,4] x1 and grad-tracked [3,4] x2 | Lt then ResetGradient(x2, false) | result stays not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Lt(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x2, false)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("[6,5,2] tensor | Lt(nil) | returns error: device mismatch", func(t *testing.T) {
			x, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Lt(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Lt tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor and tensor [1] | Lt | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Lt(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Lt tensors' dimension validation failed: expected number of dimensions to match: (0) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1] and [1,1] | Lt | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Lt(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Lt tensors' dimension validation failed: expected number of dimensions to match: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,5] | Lt | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Lt(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Lt tensors' dimension validation failed: expected number of dimensions to match: (3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1,2] and [2,1] | Lt | returns error: size mismatch at dimension 0", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Lt(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Lt tensors' dimension validation failed: expected sizes to match at dimension (0): (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,4,2] | Lt | returns error: size mismatch at dimension 1", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 4, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Lt(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Lt tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2,2], 1.) on d1 and Full([2,2], 1.) on d2 | Lt | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Lt(x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("Lt tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestLe(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar 1 | Le(scalar 1) | returns scalar 1", func(t *testing.T) {
			x1, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [0, e+eps] | Le([0, e]) | returns [1, 0]", func(t *testing.T) {
			x1, err := tensor.Of([]float64{0., math.E + 1e-15}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{0., math.E}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{1., 0.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor [[1,2,3],[-1,-2,-3]] | Le([[-1,-2,-3],[1,2,3]]) | returns [[0,0,0],[1,1,1]]", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{-1., -2., -3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{-1., -2., -3.},
				{1., 2., 3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{0., 0., 0.},
				{1., 1., 1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros tensor [1,2,3,4] | Le(ones tensor [1,2,3,4]) | returns ones tensor", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | Le(itself) | returns all ones", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Le(x)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Le([3,4] tensor) then Device() | returns the device the inputs were created on", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("two untracked [3,4] tensors | Le | y is not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor and untracked [3,4] tensor | Le | y is not gradient-tracked because operation is not differentiable", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("untracked [3,4] tensor and grad-tracked [3,4] tensor | Le | y is not gradient-tracked because operation is not differentiable", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("two grad-tracked [3,4] tensors | Le | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked [3,4] tensors | Le | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked [3,4] tensors | Le then BackPropagate | gradient is nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
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

		t.Run("two grad-tracked [3,4] tensors | Le then BackPropagate | y is still not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensors both filled with 7 | Le | returns Full([1,2^20], 1)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensors both filled with 7 | concurrent repeated Le over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x1, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x1.Le(x2)
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] x1 and untracked [3,4] x2 | Le then ResetGradient(x1, false) | result stays not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x1, false)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("untracked [3,4] x1 and grad-tracked [3,4] x2 | Le then ResetGradient(x2, false) | result stays not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Le(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x2, false)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("[6,5,2] tensor | Le(nil) | returns error: device mismatch", func(t *testing.T) {
			x, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Le(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Le tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor and tensor [1] | Le | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Le(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Le tensors' dimension validation failed: expected number of dimensions to match: (0) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1] and [1,1] | Le | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Le(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Le tensors' dimension validation failed: expected number of dimensions to match: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,5] | Le | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Le(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Le tensors' dimension validation failed: expected number of dimensions to match: (3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1,2] and [2,1] | Le | returns error: size mismatch at dimension 0", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Le(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Le tensors' dimension validation failed: expected sizes to match at dimension (0): (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,4,2] | Le | returns error: size mismatch at dimension 1", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 4, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Le(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Le tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2,2], 1.) on d1 and Full([2,2], 1.) on d2 | Le | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Le(x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("Le tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestElMax(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar 1 | ElMax(scalar 1) | returns scalar 1", func(t *testing.T) {
			x1, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [0, e+eps] | ElMax([0, e]) | returns [0, e+eps]", func(t *testing.T) {
			x1, err := tensor.Of([]float64{0., math.E + 1e-15}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{0., math.E}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{0., math.E + 1e-15}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor [[1,2,3],[-1,-2,-3]] | ElMax([[-1,-2,-3],[1,2,3]]) | returns [[1,2,3],[1,2,3]]", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{-1., -2., -3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{-1., -2., -3.},
				{1., 2., 3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{1., 2., 3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros tensor [1,2,3,4] | ElMax(ones tensor [1,2,3,4]) | returns ones tensor", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | ElMax(itself) | returns [1, 2, 3]", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.ElMax(x)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | ElMax([3,4] tensor) then Device() | returns the device the inputs were created on", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("two untracked [3,4] tensors | ElMax | y is not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor and untracked [3,4] tensor | ElMax | y is gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		t.Run("untracked [3,4] tensor and grad-tracked [3,4] tensor | ElMax | y is gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("two grad-tracked Full(nil) tensors | ElMax | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked Full(nil) tensors | ElMax | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked Full(nil) tensors | ElMax then BackPropagate | gradient is nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
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

		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked | ElMax then BackPropagate | gradient of x1 is 1, gradient of x2 is 0", func(t *testing.T) {
			x1, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full(nil, 2) and Full(nil, 2) both grad-tracked | ElMax then BackPropagate | gradient of x1 is 0.5, gradient of x2 is 0.5", func(t *testing.T) {
			x1, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full(nil, 0.5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full(nil, 0.5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("same grad-tracked scalar tensor | x.ElMax(x) then BackPropagate | gradient of x is 1 (both tie contributions accumulate)", func(t *testing.T) {
			x, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.ElMax(x)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [2,3] tensors with mixed x1>x2, x1<x2, x1==x2 | ElMax then BackPropagate | gradient is per-element pattern of 0, 1 or 0.5", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{3., 1., 2.},
				{2., 3., 1.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{1., 3., 2.},
				{2., 1., 3.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Of([][]float64{
				{1., 0., 0.5},
				{0.5, 1., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Of([][]float64{
				{0., 1., 0.5},
				{0.5, 0., 1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensors filled with 3 and 2 | ElMax | returns Full([1,2^20], 3)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large [1,2^20] grad-tracked tensors filled with 3 and 2 | ElMax then BackPropagate | gradient of x1 is Full([1,2^20], 1), gradient of x2 is Full([1,2^20], 0)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full([]int{1, n}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensors filled with 3 and 2 | concurrent repeated ElMax over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x1.ElMax(x2)
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] x1 and untracked [3,4] x2 | ElMax then ResetGradient(x1, false) | result is still gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x1, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})

		t.Run("untracked [3,4] x1 and grad-tracked [3,4] x2 | ElMax then ResetGradient(x2, false) | result is still gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMax(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x2, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("[6,5,2] tensor | ElMax(nil) | returns error: device mismatch", func(t *testing.T) {
			x, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.ElMax(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("ElMax tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor and tensor [1] | ElMax | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.ElMax(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "ElMax tensors' dimension validation failed: expected number of dimensions to match: (0) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1] and [1,1] | ElMax | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.ElMax(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "ElMax tensors' dimension validation failed: expected number of dimensions to match: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,5] | ElMax | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.ElMax(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "ElMax tensors' dimension validation failed: expected number of dimensions to match: (3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1,2] and [2,1] | ElMax | returns error: size mismatch at dimension 0", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.ElMax(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "ElMax tensors' dimension validation failed: expected sizes to match at dimension (0): (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,4,2] | ElMax | returns error: size mismatch at dimension 1", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 4, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.ElMax(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "ElMax tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2,2], 1.) on d1 and Full([2,2], 1.) on d2 | ElMax | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.ElMax(x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("ElMax tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestElMin(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar 1 | ElMin(scalar 1) | returns scalar 1", func(t *testing.T) {
			x1, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [0, e+eps] | ElMin([0, e]) | returns [0, e]", func(t *testing.T) {
			x1, err := tensor.Of([]float64{0., math.E + 1e-15}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{0., math.E}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{0., math.E}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor [[1,2,3],[-1,-2,-3]] | ElMin([[-1,-2,-3],[1,2,3]]) | returns [[-1,-2,-3],[-1,-2,-3]]", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{-1., -2., -3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{-1., -2., -3.},
				{1., 2., 3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{-1., -2., -3.},
				{-1., -2., -3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros tensor [1,2,3,4] | ElMin(ones tensor [1,2,3,4]) | returns zeros tensor", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, 2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | ElMin(itself) | returns [1, 2, 3]", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.ElMin(x)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | ElMin([3,4] tensor) then Device() | returns the device the inputs were created on", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("two untracked [3,4] tensors | ElMin | y is not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor and untracked [3,4] tensor | ElMin | y is gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		t.Run("untracked [3,4] tensor and grad-tracked [3,4] tensor | ElMin | y is gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("two grad-tracked Full(nil) tensors | ElMin | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked Full(nil) tensors | ElMin | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked Full(nil) tensors | ElMin then BackPropagate | gradient is nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
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

		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked | ElMin then BackPropagate | gradient of x1 is 0, gradient of x2 is 1", func(t *testing.T) {
			x1, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full(nil, 2) and Full(nil, 2) both grad-tracked | ElMin then BackPropagate | gradient of x1 is 0.5, gradient of x2 is 0.5", func(t *testing.T) {
			x1, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full(nil, 0.5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full(nil, 0.5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("same grad-tracked scalar tensor | x.ElMin(x) then BackPropagate | gradient of x is 1 (both tie contributions accumulate)", func(t *testing.T) {
			x, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.ElMin(x)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [2,3] tensors with mixed x1<x2, x1>x2, x1==x2 | ElMin then BackPropagate | gradient is per-element pattern of 0, 1 or 0.5", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{1., 3., 2.},
				{2., 1., 3.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{3., 1., 2.},
				{2., 3., 1.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Of([][]float64{
				{1., 0., 0.5},
				{0.5, 1., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Of([][]float64{
				{0., 1., 0.5},
				{0.5, 0., 1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensors filled with 3 and 2 | ElMin | returns Full([1,2^20], 2)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large [1,2^20] grad-tracked tensors filled with 3 and 2 | ElMin then BackPropagate | gradient of x1 is Full([1,2^20], 0), gradient of x2 is Full([1,2^20], 1)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full([]int{1, n}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensors filled with 3 and 2 | concurrent repeated ElMin over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x1.ElMin(x2)
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] x1 and untracked [3,4] x2 | ElMin then ResetGradient(x1, false) | result is still gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x1, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})

		t.Run("untracked [3,4] x1 and grad-tracked [3,4] x2 | ElMin then ResetGradient(x2, false) | result is still gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.ElMin(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x2, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("[6,5,2] tensor | ElMin(nil) | returns error: device mismatch", func(t *testing.T) {
			x, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.ElMin(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("ElMin tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor and tensor [1] | ElMin | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.ElMin(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "ElMin tensors' dimension validation failed: expected number of dimensions to match: (0) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1] and [1,1] | ElMin | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.ElMin(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "ElMin tensors' dimension validation failed: expected number of dimensions to match: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,5] | ElMin | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.ElMin(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "ElMin tensors' dimension validation failed: expected number of dimensions to match: (3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1,2] and [2,1] | ElMin | returns error: size mismatch at dimension 0", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.ElMin(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "ElMin tensors' dimension validation failed: expected sizes to match at dimension (0): (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [6,5,2] and [6,4,2] | ElMin | returns error: size mismatch at dimension 1", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 4, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.ElMin(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "ElMin tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2,2], 1.) on d1 and Full([2,2], 1.) on d2 | ElMin | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.ElMin(x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("ElMin tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestAdd(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar 4 | Add(scalar 2) | returns scalar 6", func(t *testing.T) {
			x1, err := tensor.Of(4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(6., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | Add([4, 5, 6]) | returns [5, 7, 9]", func(t *testing.T) {
			x1, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{4., 5., 6.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{5., 7., 9.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D tensor | Add(3D tensor) | returns element-wise sums", func(t *testing.T) {
			x1, err := tensor.Of([][][]float64{
				{{-5.}, {1.}},
				{{-9.}, {2.}},
				{{2.}, {-1.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][][]float64{
				{{-4.}, {7.}},
				{{6.}, {2.}},
				{{-3.}, {4.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
				{{-9.}, {8.}},
				{{-3.}, {4.}},
				{{-1.}, {3.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros [3,1,5,1] | Add(ones [1,2,3,4,1,6]) | returns ones [1,2,3,4,5,6] (broadcast)", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 1, 5, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4, 1, 6}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, 2, 3, 4, 5, 6}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | Add(itself) | returns [2, 4, 6]", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Add(x)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{2., 4., 6.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Add([3,4] tensor) then Device() | returns the device the inputs were created on", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("two untracked [3,4] tensors | Add | y is not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor and untracked [3,4] tensor | Add | y is gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		t.Run("untracked [3,4] tensor and grad-tracked [3,4] tensor | Add | y is gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("two grad-tracked Full(nil) tensors | Add | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked Full(nil) tensors | Add | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked Full(nil) tensors | Add then BackPropagate | gradient is nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
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

		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked | Add then BackPropagate | gradient of x1 is 1, gradient of x2 is 1", func(t *testing.T) {
			x1, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("same grad-tracked scalar tensor | x.Add(x) then BackPropagate | gradient of x is 2", func(t *testing.T) {
			x, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Add(x)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full(nil, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("two grad-tracked [2,3] tensors | Add then BackPropagate | gradient of each is all-ones [2,3]", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensors filled with 3 and 2 | Add | returns Full([1,2^20], 5)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large [1,2^20] grad-tracked tensors filled with 3 and 2 | Add then BackPropagate | gradient of each is Full([1,2^20], 1)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensors filled with 3 and 2 | concurrent repeated Add over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x1.Add(x2)
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] x1 and untracked [3,4] x2 | Add then ResetGradient(x1, false) | result is still gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x1, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})

		t.Run("untracked [3,4] x1 and grad-tracked [3,4] x2 | Add then ResetGradient(x2, false) | result is still gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x2, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("[1,5,2,4,1] tensor | Add(nil) | returns error: device mismatch", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 5, 2, 4, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Add(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Add tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [2] and [3] | Add | returns error: broadcast incompatibility at dimension 0 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Add(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Add tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [3] and [2] | Add | returns error: broadcast incompatibility at dimension 0 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Add(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Add tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [4,2,1] and [4,3,5] | Add | returns error: broadcast incompatibility at dimension 1 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{4, 3, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Add(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Add tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [4,3,5] and [4,2,1] | Add | returns error: broadcast incompatibility at dimension 1 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 3, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{4, 2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Add(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Add tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [2,1,4,1] and [2,3,4,4,5] | Add | returns error: broadcast incompatibility at dimension 1 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 1, 4, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 3, 4, 4, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Add(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Add tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [2,3,4,4,5] and [2,1,4,1] | Add | returns error: broadcast incompatibility at dimension 1 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3, 4, 4, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1, 4, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Add(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Add tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [3,1,3,6] and [1,2,3,4,5,6] | Add | returns error: broadcast incompatibility at dimension 4 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 1, 3, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4, 5, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Add(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (4)")
			} else if err.Error() != "Add tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (3) or source size to be (1) at dimension (4): got shape (5)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1,2,3,4,5,6] and [3,1,3,6] | Add | returns error: broadcast incompatibility at dimension 4 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2, 3, 4, 5, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 1, 3, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Add(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (4)")
			} else if err.Error() != "Add tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (3) or source size to be (1) at dimension (4): got shape (5)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2,2], 1.) on d1 and Full([2,2], 1.) on d2 | Add | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Add(x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("Add tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestSub(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar 4 | Sub(scalar 2) | returns scalar 2", func(t *testing.T) {
			x1, err := tensor.Of(4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | Sub([4, 5, 6]) | returns [-3, -3, -3]", func(t *testing.T) {
			x1, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{4., 5., 6.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{-3., -3., -3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D tensor | Sub(3D tensor) | returns element-wise differences", func(t *testing.T) {
			x1, err := tensor.Of([][][]float64{
				{{-5.}, {1.}},
				{{-9.}, {2.}},
				{{2.}, {-1.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][][]float64{
				{{-4.}, {7.}},
				{{6.}, {2.}},
				{{-3.}, {4.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
				{{-1.}, {-6.}},
				{{-15.}, {0.}},
				{{5.}, {-5.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("ones [3,1,5,1] | Sub(zeros [1,2,3,4,1,6]) | returns ones [1,2,3,4,5,6] (broadcast)", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 1, 5, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4, 1, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, 2, 3, 4, 5, 6}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | Sub(itself) | returns zeros tensor", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Sub(x)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Sub([3,4] tensor) then Device() | returns the device the inputs were created on", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("two untracked [3,4] tensors | Sub | y is not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor and untracked [3,4] tensor | Sub | y is gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		t.Run("untracked [3,4] tensor and grad-tracked [3,4] tensor | Sub | y is gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("two grad-tracked Full(nil) tensors | Sub | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked Full(nil) tensors | Sub | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked Full(nil) tensors | Sub then BackPropagate | gradient is nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
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

		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked | Sub then BackPropagate | gradient of x1 is 1, gradient of x2 is -1", func(t *testing.T) {
			x1, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full(nil, -1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("same grad-tracked scalar tensor | x.Sub(x) then BackPropagate | gradient of x is 0", func(t *testing.T) {
			x, err := tensor.Full(nil, 5., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Sub(x)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("two grad-tracked [2,3] tensors | Sub then BackPropagate | gradient of x1 is all-ones [2,3], gradient of x2 is all-neg-ones [2,3]", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full([]int{2, 3}, -1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensors filled with 3 and 2 | Sub | returns Full([1,2^20], 1)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large [1,2^20] grad-tracked tensors filled with 3 and 2 | Sub then BackPropagate | gradient of x1 is Full([1,2^20], 1), gradient of x2 is Full([1,2^20], -1)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full([]int{1, n}, -1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensors filled with 3 and 2 | concurrent repeated Sub over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x1.Sub(x2)
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] x1 and untracked [3,4] x2 | Sub then ResetGradient(x1, false) | result is still gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x1, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})

		t.Run("untracked [3,4] x1 and grad-tracked [3,4] x2 | Sub then ResetGradient(x2, false) | result is still gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Sub(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x2, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("[1,5,2,4,1] tensor | Sub(nil) | returns error: device mismatch", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 5, 2, 4, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Sub(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Sub tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [2] and [3] | Sub | returns error: broadcast incompatibility at dimension 0 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Sub(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Sub tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [3] and [2] | Sub | returns error: broadcast incompatibility at dimension 0 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Sub(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Sub tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [4,2,1] and [4,3,5] | Sub | returns error: broadcast incompatibility at dimension 1 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{4, 3, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Sub(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Sub tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [4,3,5] and [4,2,1] | Sub | returns error: broadcast incompatibility at dimension 1 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 3, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{4, 2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Sub(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Sub tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [2,1,4,1] and [2,3,4,4,5] | Sub | returns error: broadcast incompatibility at dimension 1 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 1, 4, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 3, 4, 4, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Sub(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Sub tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [2,3,4,4,5] and [2,1,4,1] | Sub | returns error: broadcast incompatibility at dimension 1 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3, 4, 4, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1, 4, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Sub(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Sub tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [3,1,3,6] and [1,2,3,4,5,6] | Sub | returns error: broadcast incompatibility at dimension 4 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 1, 3, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4, 5, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Sub(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (4)")
			} else if err.Error() != "Sub tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (3) or source size to be (1) at dimension (4): got shape (5)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1,2,3,4,5,6] and [3,1,3,6] | Sub | returns error: broadcast incompatibility at dimension 4 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2, 3, 4, 5, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 1, 3, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Sub(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (4)")
			} else if err.Error() != "Sub tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (3) or source size to be (1) at dimension (4): got shape (5)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2,2], 1.) on d1 and Full([2,2], 1.) on d2 | Sub | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Sub(x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("Sub tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestMul(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar 4 | Mul(scalar 2) | returns scalar 8", func(t *testing.T) {
			x1, err := tensor.Of(4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Mul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(8., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | Mul([4, 5, 6]) | returns [4, 10, 18]", func(t *testing.T) {
			x1, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{4., 5., 6.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Mul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{4., 10., 18.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D tensor | Mul(3D tensor) | returns element-wise products", func(t *testing.T) {
			x1, err := tensor.Of([][][]float64{
				{{-5.}, {1.}},
				{{-9.}, {2.}},
				{{2.}, {-1.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][][]float64{
				{{-4.}, {7.}},
				{{6.}, {2.}},
				{{-3.}, {4.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Mul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
				{{20.}, {7.}},
				{{-54.}, {4.}},
				{{-6.}, {-4.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("ones [3,1,5,1] | Mul(zeros [1,2,3,4,1,6]) | broadcasts to zeros [1,2,3,4,5,6]", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 1, 5, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4, 1, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Mul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, 2, 3, 4, 5, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | Mul(itself) | returns [1, 4, 9]", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Mul(x)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{1., 4., 9.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// =============== gradients ===============

		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked | Mul then BackPropagate | gradient of x1 is 2, gradient of x2 is 3", func(t *testing.T) {
			x1, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Mul(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,2]{1,2,3,4}) grad-tracked | x.Mul(x) then BackPropagate | gradient of x is 2x element-wise", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{1., 2.},
				{3., 4.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Mul(x)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Of([][]float64{
				{2., 4.},
				{6., 8.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full(nil, 2) grad-tracked | x.Mul(x).Mul(x) then BackPropagate | gradient of x is 3x²=12", func(t *testing.T) {
			x, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			tmp, err := x.Mul(x)
			if err != nil {
				t.Fatal(err)
			}
			y, err := tmp.Mul(x)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full(nil, 12., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("x1 untracked, x2 grad-tracked | Mul then BackPropagate | gradient of y is non-nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Mul(x2)
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

		t.Run("x1 grad-tracked, x2 untracked | Mul then BackPropagate | gradient of y is non-nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Mul(x2)
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

		t.Run("x1 untracked, x2 untracked | Mul then BackPropagate | gradient of y is nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Mul(x2)
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

		// ============================== extra functionalities ==============================

		// ============================== side effects ==============================

		// ============================== validations ==============================

		t.Run("nil input tensor | Mul(nil) | returns error: device mismatch", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 5, 2, 4, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Mul(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Mul tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [2] and [3] | Mul | returns error: broadcast incompatibility at dimension 0 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Mul(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Mul tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [3] and [2] | Mul | returns error: broadcast incompatibility at dimension 0 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Mul(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Mul tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [4,2,1] and [4,3,5] | Mul | returns error: broadcast incompatibility at dimension 1 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{4, 3, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Mul(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Mul tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [4,3,5] and [4,2,1] | Mul | returns error: broadcast incompatibility at dimension 1 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 3, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{4, 2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Mul(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Mul tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [2,1,4,1] and [2,3,4,4,5] | Mul | returns error: broadcast incompatibility at dimension 1 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 1, 4, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 3, 4, 4, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Mul(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Mul tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [2,3,4,4,5] and [2,1,4,1] | Mul | returns error: broadcast incompatibility at dimension 1 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3, 4, 4, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1, 4, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Mul(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Mul tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [3,1,3,6] and [1,2,3,4,5,6] | Mul | returns error: broadcast incompatibility at dimension 4 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 1, 3, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4, 5, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Mul(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (4)")
			} else if err.Error() != "Mul tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (3) or source size to be (1) at dimension (4): got shape (5)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1,2,3,4,5,6] and [3,1,3,6] | Mul | returns error: broadcast incompatibility at dimension 4 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2, 3, 4, 5, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 1, 3, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Mul(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (4)")
			} else if err.Error() != "Mul tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (3) or source size to be (1) at dimension (4): got shape (5)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2,2], 1.) on d1 and Full([2,2], 1.) on d2 | Mul | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Mul(x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("Mul tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestDiv(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar 4 | Div(scalar 2) | returns scalar 2", func(t *testing.T) {
			x1, err := tensor.Of(4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Div(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [4, 5, 6] | Div([1, 2, 3]) | returns [4, 2.5, 2]", func(t *testing.T) {
			x1, err := tensor.Of([]float64{4., 5., 6.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Div(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{4., 2.5, 2.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D tensor | Div(3D tensor) | returns element-wise quotients", func(t *testing.T) {
			x1, err := tensor.Of([][][]float64{
				{{0.}, {1.}},
				{{2.}, {3.}},
				{{4.}, {5.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][][]float64{
				{{-1.}, {1.}},
				{{-2.}, {2.}},
				{{-4.}, {5.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Div(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
				{{0.}, {1.}},
				{{-1.}, {1.5}},
				{{-1.}, {1.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("zeros [3,1,5,1] | Div(ones [1,2,3,4,1,6]) | broadcasts to zeros [1,2,3,4,5,6]", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 1, 5, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4, 1, 6}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Div(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, 2, 3, 4, 5, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1, 2, 3] | Div(itself) | returns all ones", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Div(x)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// =============== gradients ===============

		t.Run("Full(nil, 3) and Full(nil, 2) both grad-tracked | Div then BackPropagate | gradient of x1 is 0.5, gradient of x2 is -0.75", func(t *testing.T) {
			x1, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Div(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full(nil, 0.5, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full(nil, -0.75, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,2]{1,2,3,4}) grad-tracked | x.Div(x) then BackPropagate | gradient of x is all-zeros", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{1., 2.},
				{3., 4.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Div(x)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full([]int{2, 2}, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full(nil, 2) grad-tracked | x.Div(x.Pow(2)) then BackPropagate | gradient of x is -1/x²=-0.25", func(t *testing.T) {
			x, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Div(x.Pow(2.))
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full(nil, -0.25, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("x1 untracked, x2 grad-tracked | Div then BackPropagate | gradient of y is non-nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Div(x2)
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

		t.Run("x1 grad-tracked, x2 untracked | Div then BackPropagate | gradient of y is non-nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Div(x2)
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

		t.Run("x1 untracked, x2 untracked | Div then BackPropagate | gradient of y is nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Div(x2)
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

		// ============================== extra functionalities ==============================

		// ============================== side effects ==============================

		// ============================== validations ==============================

		t.Run("nil input tensor | Div(nil) | returns error: device mismatch", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 5, 2, 4, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Div(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Div tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [2] and [3] | Div | returns error: broadcast incompatibility at dimension 0 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Div(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Div tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [3] and [2] | Div | returns error: broadcast incompatibility at dimension 0 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Div(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Div tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [4,2,1] and [4,3,5] | Div | returns error: broadcast incompatibility at dimension 1 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{4, 3, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Div(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Div tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [4,3,5] and [4,2,1] | Div | returns error: broadcast incompatibility at dimension 1 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 3, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{4, 2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Div(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Div tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [2,1,4,1] and [2,3,4,4,5] | Div | returns error: broadcast incompatibility at dimension 1 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 1, 4, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 3, 4, 4, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Div(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Div tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [2,3,4,4,5] and [2,1,4,1] | Div | returns error: broadcast incompatibility at dimension 1 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3, 4, 4, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1, 4, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Div(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Div tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [3,1,3,6] and [1,2,3,4,5,6] | Div | returns error: broadcast incompatibility at dimension 4 of first operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 1, 3, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2, 3, 4, 5, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Div(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (4)")
			} else if err.Error() != "Div tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (3) or source size to be (1) at dimension (4): got shape (5)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("tensors [1,2,3,4,5,6] and [3,1,3,6] | Div | returns error: broadcast incompatibility at dimension 4 of second operand", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2, 3, 4, 5, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 1, 3, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Div(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (4)")
			} else if err.Error() != "Div tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (3) or source size to be (1) at dimension (4): got shape (5)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2,2], 1.) on d1 and Full([2,2], 1.) on d2 | Div | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Div(x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("Div tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestDot(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("1D tensor [5] and 1D tensor [2] | Dot | returns scalar 10", func(t *testing.T) {
			x1, err := tensor.Of([]float64{5.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{2.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Dot(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(10., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [0, 1, 2] and 1D tensor [3, 4, 5] | Dot | returns scalar 14", func(t *testing.T) {
			x1, err := tensor.Of([]float64{0., 1., 2.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{3., 4., 5.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Dot(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(14., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D ones tensor [5,1] and 2D ones tensor [5,1] | Dot | returns 1D ones tensor [5]", func(t *testing.T) {
			x1, err := tensor.Full([]int{5, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{5, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Dot(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{5}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D ones tensor [5,4] and 2D ones tensor [5,4] | Dot | returns 1D tensor [5] filled with 4", func(t *testing.T) {
			x1, err := tensor.Full([]int{5, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{5, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Dot(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{5}, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("5D ones tensor [4,5,1,7,8] and 6D ones tensor [3,4,1,6,1,8] | Dot | returns 5D tensor [3,4,5,6,7] filled with 8", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 5, 1, 7, 8}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4, 1, 6, 1, 8}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Dot(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3, 4, 5, 6, 7}, 8., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("4D tensor [2,3,1,3] and 2D tensor [4,3] | Dot | returns 3D tensor [2,3,4] via broadcasting", func(t *testing.T) {
			x1, err := tensor.Of([][][][]float64{
				{
					{{1., 1., 1.}},
					{{1., 1., 1.}},
					{{1., 1., 1.}},
				},
				{
					{{1., 1., 1.}},
					{{1., 1., 1.}},
					{{1., 1., 1.}},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
				{-1., 0., 1.},
				{2., -2., 2.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Dot(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
				{
					{6., 15., 0., 2.},
					{6., 15., 0., 2.},
					{6., 15., 0., 2.},
				},
				{
					{6., 15., 0., 2.},
					{6., 15., 0., 2.},
					{6., 15., 0., 2.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// =============== gradients ===============

		t.Run("Full([2], 2) and Full([2], 3) both grad-tracked | Dot then BackPropagate | gradient of x1 is Full([2], 3), gradient of x2 is Full([2], 2)", func(t *testing.T) {
			x1, err := tensor.Full([]int{2}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Dot(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full([]int{2}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full([]int{2}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([3]{1,2,3}) grad-tracked | x.Dot(x) then BackPropagate | gradient of x is 2x=[2,4,6]", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Dot(x)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Of([]float64{2., 4., 6.}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([4]{1,2,3,4}) and Of([4]{4,3,2,1}) both grad-tracked | Dot then BackPropagate | gradient of x1 is x2, gradient of x2 is x1", func(t *testing.T) {
			x1, err := tensor.Of([]float64{1., 2., 3., 4.}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{4., 3., 2., 1.}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Dot(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Of([]float64{4., 3., 2., 1.}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Of([]float64{1., 2., 3., 4.}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("x1 untracked, x2 grad-tracked | Dot then BackPropagate | gradient of y is non-nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Dot(x2)
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

		t.Run("x1 grad-tracked, x2 untracked | Dot then BackPropagate | gradient of y is non-nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Dot(x2)
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

		t.Run("x1 untracked, x2 untracked | Dot then BackPropagate | gradient of y is nil", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Dot(x2)
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

		// ============================== extra functionalities ==============================

		// ============================== side effects ==============================

		// ============================== validations ==============================

		t.Run("nil input tensor | Dot(nil) | returns error: device mismatch", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Dot(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Dot tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("0D tensor and 0D tensor | Dot | returns error: tensors must have at least 1 dimension", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Dot(x)
			if err == nil {
				t.Fatal("expected error because of tensors having less than 1 dimension")
			} else if err.Error() != "Dot tensors' dimension validation failed: expected tensors to have at least (1) dimension for dot product: got (0) and (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("0D tensor and 1D tensor | Dot | returns error: first tensor must have at least 1 dimension", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Dot(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having less than 1 dimension")
			} else if err.Error() != "Dot tensors' dimension validation failed: expected tensors to have at least (1) dimension for dot product: got (0) and (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("1D tensor and 0D tensor | Dot | returns error: second tensor must have at least 1 dimension", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x2.Dot(x1)
			if err == nil {
				t.Fatal("expected error because of tensors having less than 1 dimension")
			} else if err.Error() != "Dot tensors' dimension validation failed: expected tensors to have at least (1) dimension for dot product: got (1) and (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("1D tensor of size 3 and 1D tensor of size 2 | Dot | returns error: last dimensions do not match", func(t *testing.T) {
			x1, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Dot(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at last dimension")
			} else if err.Error() != "Dot tensors' dimension validation failed: expected sizes to match at last dimensions: (3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("2D tensor [8,4] and 3D tensor [5,2,4] | Dot | returns error: batch dimensions not broadcastable", func(t *testing.T) {
			x1, err := tensor.Full([]int{8, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{5, 2, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Dot(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Dot tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (8)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("3D tensor [5,2,4] and 2D tensor [8,4] | Dot | returns error: first operand batch dimensions not broadcastable", func(t *testing.T) {
			x1, err := tensor.Full([]int{5, 2, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{8, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Dot(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Dot tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (1): got shape (8)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("3D tensor [2,3,4] and 3D tensor [3,3,4] | Dot | returns error: leading batch dimensions not broadcastable", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Dot(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Dot tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("5D tensor [4,1,3,5,2] and 5D tensor [4,6,2,5,2] | Dot | returns error: middle batch dimensions not broadcastable", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 1, 3, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{4, 6, 2, 5, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Dot(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (2)")
			} else if err.Error() != "Dot tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (2): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("3D tensor [2,3,4] and 3D tensor [3,2,4] | Dot | returns error: both operands not broadcastable: first operand is reported", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 2, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Dot(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0) for both operands")
			} else if err.Error() != "Dot tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("2D tensor [2,3] and 2D tensor [2,4] | Dot | returns error: last dimensions do not match", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Dot(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at last dimension")
			} else if err.Error() != "Dot tensors' dimension validation failed: expected sizes to match at last dimensions: (3) != (4)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("1D tensor [3] and 2D tensor [2,4] | Dot | returns error: last dimensions do not match", func(t *testing.T) {
			x1, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Dot(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at last dimension")
			} else if err.Error() != "Dot tensors' dimension validation failed: expected sizes to match at last dimensions: (3) != (4)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("3D tensor [2,3,4] and 3D tensor [3,3,5] | Dot | returns error: last dimensions error takes precedence over broadcasting error", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 3, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Dot(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at last dimension")
			} else if err.Error() != "Dot tensors' dimension validation failed: expected sizes to match at last dimensions: (4) != (5)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2,2], 1.) on d1 and Full([2,2], 1.) on d2 | Dot | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Dot(x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("Dot tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestMatMul(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("2D tensor [[5]] | MatMul([[2]]) | returns [[10]]", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{{5.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{{2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{{10.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor [[8],[4]] | MatMul([[2]]) | returns [[16],[8]]", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{{8.}, {4.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{{2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{{16.}, {8.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor [[2]] | MatMul([[8,4]]) | returns [[16,8]]", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{{2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{{8., 4.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{{16., 8.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor [[0,1],[2,3]] | MatMul([[4,5],[6,7]]) | returns [[6,7],[26,31]]", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{0., 1.},
				{2., 3.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{4., 5.},
				{6., 7.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{6., 7.},
				{26., 31.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor [2x3] of ones | MatMul([3x2] of 3,4,5) | returns [2x2] result", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{1., 1., 1.},
				{2., 2., 2.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{3., 3.},
				{4., 4.},
				{5., 5.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{12., 12.},
				{24., 24.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D tensor [2,2,2] | MatMul([2,2,2]) | returns batched 2x2 matrix products", func(t *testing.T) {
			x1, err := tensor.Of([][][]float64{
				{
					{1., 0.},
					{2., 1.},
				},
				{
					{3., 1.},
					{0., 2.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][][]float64{
				{
					{2., 3.},
					{0., 1.},
				},
				{
					{1., 2.},
					{3., 0.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
				{
					{2., 3.},
					{4., 7.},
				},
				{
					{6., 6.},
					{6., 0.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("4D tensor [1,2,3,4] | MatMul([1,2,4,2]) | returns batched 3x2 matrix products", func(t *testing.T) {
			x1, err := tensor.Of([][][][]float64{
				{
					{
						{1., 2., 3., 4.},
						{5., 6., 7., 8.},
						{9., 10., 11., 12.},
					},
					{
						{1., 0., 0., 0.},
						{0., 1., 0., 0.},
						{0., 0., 1., 0.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][][][]float64{
				{
					{
						{1., 0.},
						{0., 1.},
						{1., 0.},
						{0., 1.},
					},
					{
						{1., 2.},
						{3., 4.},
						{5., 6.},
						{7., 8.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][][]float64{
				{
					{
						{4., 6.},
						{12., 14.},
						{20., 22.},
					},
					{
						{1., 2.},
						{3., 4.},
						{5., 6.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D ones tensor [5,1,1] | MatMul([5,1,1] ones) | returns [5,1,1] ones", func(t *testing.T) {
			x1, err := tensor.Full([]int{5, 1, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{5, 1, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{5, 1, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D ones tensor [4,2,1] | MatMul([4,1,1] ones) | returns [4,2,1] ones", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 2, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{4, 1, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{4, 2, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D ones tensor [4,1,1] | MatMul([4,1,2] ones) | returns [4,1,2] ones", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 1, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{4, 1, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{4, 1, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("4D ones tensor [5,4,2,2] | MatMul([5,4,2,2] ones) | returns [5,4,2,2] tensor filled with 2", func(t *testing.T) {
			x1, err := tensor.Full([]int{5, 4, 2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{5, 4, 2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{5, 4, 2, 2}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("6D ones tensor [7,6,1,4,3,2] | MatMul([1,5,1,2,3] ones) | returns [7,6,5,4,3,3] tensor filled with 2", func(t *testing.T) {
			x1, err := tensor.Full([]int{7, 6, 1, 4, 3, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 5, 1, 2, 3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{7, 6, 5, 4, 3, 3}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor [2,3] | MatMul(4D tensor [1,2,3,2]) | broadcasts and returns [1,2,2,2]", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][][][]float64{
				{
					{
						{1., 0.},
						{0., 1.},
						{1., 1.},
					},
					{
						{2., 0.},
						{0., 2.},
						{1., 1.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][][]float64{
				{
					{
						{4., 5.},
						{10., 11.},
					},
					{
						{5., 7.},
						{14., 16.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// =============== gradients ===============

		t.Run("Full([2,3], 2) and Full([3,2], 3) both grad-tracked | MatMul then BackPropagate | gradient of x1 is Full([2,3], 6), gradient of x2 is Full([3,2], 4)", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 2}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full([]int{2, 3}, 6., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full([]int{3, 2}, 4., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Identity [3,3] grad-tracked | x.MatMul(x) then BackPropagate | gradient of x is Full([3,3], 2)", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{1., 0., 0.},
				{0., 1., 0.},
				{0., 0., 1.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MatMul(x)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full([]int{3, 3}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([2,3],1) grad-tracked, Full([3,2],1) and Full([2,2],0) untracked | x1.MatMul(x2).Add(x3) then BackPropagate | gradient of x1 is Full([2,3], 2)", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x3, err := tensor.Full([]int{2, 2}, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			mm, err := x1.MatMul(x2)
			if err != nil {
				t.Fatal(err)
			}
			y, err := mm.Add(x3)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x1.Gradient()

			h, err := tensor.Full([]int{2, 3}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("x1 untracked, x2 grad-tracked | MatMul then BackPropagate | gradient of y is non-nil", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{{1.}}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{{1.}}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
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

		t.Run("x1 grad-tracked, x2 untracked | MatMul then BackPropagate | gradient of y is non-nil", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{{1.}}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{{1.}}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
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

		t.Run("x1 untracked, x2 untracked | MatMul then BackPropagate | gradient of y is nil", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{{1.}}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{{1.}}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.MatMul(x2)
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

		// ============================== extra functionalities ==============================

		// ============================== side effects ==============================

		// ============================== validations ==============================

		t.Run("nil input tensor | MatMul(nil) | returns error: device mismatch", func(t *testing.T) {
			x, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.MatMul(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("MatMul tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("1D tensor of size 3 and 1D tensor of size 3 | MatMul | returns error: tensors must have at least 2 dimensions", func(t *testing.T) {
			x, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.MatMul(x)
			if err == nil {
				t.Fatal("expected error because of tensors having less than (2) dimensions")
			} else if err.Error() != "MatMul tensors' dimension validation failed: expected tensors to have at least (2) dimensions for matrix multiplication: got (1) and (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("1D tensor of size 3 and 2D tensor [2,3] | MatMul | returns error: first tensor must have at least 2 dimensions", func(t *testing.T) {
			x1, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.MatMul(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having less than (2) dimensions")
			} else if err.Error() != "MatMul tensors' dimension validation failed: expected tensors to have at least (2) dimensions for matrix multiplication: got (1) and (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("2D tensor [2,3] and 1D tensor of size 3 | MatMul | returns error: second tensor must have at least 2 dimensions", func(t *testing.T) {
			x1, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x2.MatMul(x1)
			if err == nil {
				t.Fatal("expected error because of tensors having less than (2) dimensions")
			} else if err.Error() != "MatMul tensors' dimension validation failed: expected tensors to have at least (2) dimensions for matrix multiplication: got (2) and (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("2D tensor [3,3] and 2D tensor [2,3] | MatMul | returns error: inner dimensions do not match", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.MatMul(x2)
			if err == nil {
				t.Fatal("expected error because of size incompatiblity in the last 2 dimensions for matrix multiplication")
			} else if err.Error() != "MatMul tensors' dimension validation failed: expected dimension (1) of first tensor to be equal to dimension (0) of second tensor for matrix multiplication: (3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("4D tensor [5,5,3,2] and 3D tensor [5,3,3] | MatMul | returns error: inner dimensions do not match", func(t *testing.T) {
			x1, err := tensor.Full([]int{5, 5, 3, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{5, 3, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.MatMul(x2)
			if err == nil {
				t.Fatal("expected error because of size incompatiblity in the last 2 dimensions for matrix multiplication")
			} else if err.Error() != "MatMul tensors' dimension validation failed: expected dimension (3) of first tensor to be equal to dimension (1) of second tensor for matrix multiplication: (2) != (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("3D tensor [3,4,3] and 4D tensor [5,6,3,4] | MatMul | returns error: first tensor batch dimensions not broadcastable", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{5, 6, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.MatMul(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "MatMul tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (3) or source size to be (1) at dimension (1): got shape (6)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("3D tensor [6,4,3] and 5D tensor [6,2,5,3,4] | MatMul | returns error: batch dimensions not broadcastable", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 4, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 2, 5, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.MatMul(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (2)")
			} else if err.Error() != "MatMul tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (5) or source size to be (1) at dimension (2): got shape (6)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("0D tensor and 0D tensor | MatMul | returns error: tensors must have at least 2 dimensions", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.MatMul(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having less than (2) dimensions")
			} else if err.Error() != "MatMul tensors' dimension validation failed: expected tensors to have at least (2) dimensions for matrix multiplication: got (0) and (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("0D tensor and 2D tensor [2,2] | MatMul | returns error: first tensor must have at least 2 dimensions", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.MatMul(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having less than (2) dimensions")
			} else if err.Error() != "MatMul tensors' dimension validation failed: expected tensors to have at least (2) dimensions for matrix multiplication: got (0) and (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("2D tensor [2,2] and 0D tensor | MatMul | returns error: second tensor must have at least 2 dimensions", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.MatMul(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having less than (2) dimensions")
			} else if err.Error() != "MatMul tensors' dimension validation failed: expected tensors to have at least (2) dimensions for matrix multiplication: got (2) and (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("3D tensor [2,3,4] and 3D tensor [3,4,5] | MatMul | returns error: first tensor leading batch dimension not broadcastable", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.MatMul(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "MatMul tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("3D tensor [3,3,4] and 3D tensor [2,4,5] | MatMul | returns error: second tensor leading batch dimension not broadcastable", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 4, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.MatMul(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "MatMul tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("6D tensor [2,1,3,5,4,3] and 6D tensor [2,6,2,5,3,4] | MatMul | returns error: middle batch dimensions not broadcastable", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 1, 3, 5, 4, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 6, 2, 5, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.MatMul(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (2)")
			} else if err.Error() != "MatMul tensors' broadcasting failed: failed to broadcast second operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (2): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("4D tensor [2,3,4,5] and 4D tensor [3,2,5,6] | MatMul | returns error: both operands not broadcastable: first operand is reported", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3, 4, 5}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 2, 5, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.MatMul(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0) for both operands")
			} else if err.Error() != "MatMul tensors' broadcasting failed: failed to broadcast first operand: Broadcast input shape validation failed: expected target shape to be (2) or source size to be (1) at dimension (0): got shape (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("3D tensor [2,3,4] and 3D tensor [3,5,6] | MatMul | returns error: inner dimensions error takes precedence over broadcasting error", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 5, 6}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.MatMul(x2)
			if err == nil {
				t.Fatal("expected error because of size incompatiblity in the inner dimensions")
			} else if err.Error() != "MatMul tensors' dimension validation failed: expected dimension (2) of first tensor to be equal to dimension (1) of second tensor for matrix multiplication: (4) != (5)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2,2], 1.) on d1 and Full([2,2], 1.) on d2 | MatMul | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.MatMul(x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("MatMul tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestPatch(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("Full(nil, 0.) scalar and Full(nil, 1.) scalar | Patch(nil) | replaces scalar value", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch(nil, x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([2], 0.) and Full([1], 1.) | Patch([{1,2}]) | patches last element", func(t *testing.T) {
			x1, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch([]tensor.Range{{From: 1, To: 2}}, x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{0., 1.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,3], 0.) and Full([2,2], 1.) | Patch([{0,2},{0,2}]) | patches top-left 2x2 block", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch([]tensor.Range{{From: 0, To: 2}, {From: 0, To: 2}}, x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{1., 1., 0.},
				{1., 1., 0.},
				{0., 0., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,3], 0.) and Full([2,2], 1.) | Patch([{0,2},{1,3}]) | patches top-right 2x2 block", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch([]tensor.Range{{From: 0, To: 2}, {From: 1, To: 3}}, x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{0., 1., 1.},
				{0., 1., 1.},
				{0., 0., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,3], 0.) and Full([2,2], 1.) | Patch([{1,3},{0,2}]) | patches bottom-left 2x2 block", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch([]tensor.Range{{From: 1, To: 3}, {From: 0, To: 2}}, x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{0., 0., 0.},
				{1., 1., 0.},
				{1., 1., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,3], 0.) and Full([2,2], 1.) | Patch([{1,3},{1,3}]) | patches bottom-right 2x2 block", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch([]tensor.Range{{From: 1, To: 3}, {From: 1, To: 3}}, x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{0., 0., 0.},
				{0., 1., 1.},
				{0., 1., 1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([4,3,2], 0.) and Full([3,2,1], 2.) | Patch(nil) | patches from origin with nil ranges", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 3, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 2, 1}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch(nil, x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
				{
					{2., 0.},
					{2., 0.},
					{0., 0.},
				},
				{
					{2., 0.},
					{2., 0.},
					{0., 0.},
				},
				{
					{2., 0.},
					{2., 0.},
					{0., 0.},
				},
				{
					{0., 0.},
					{0., 0.},
					{0., 0.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([4,3,2], 0.) and Full([3,2,1], 3.) | Patch([{1,4}]) | patches from dim-0 offset 1", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 3, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 2, 1}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch([]tensor.Range{{From: 1, To: 4}}, x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
				{
					{0., 0.},
					{0., 0.},
					{0., 0.},
				},
				{
					{3., 0.},
					{3., 0.},
					{0., 0.},
				},
				{
					{3., 0.},
					{3., 0.},
					{0., 0.},
				},
				{
					{3., 0.},
					{3., 0.},
					{0., 0.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([4,3,2], 0.) and Full([3,2,1], 4.) | Patch([{1,4},{1,3},{1,2}]) | patches from all-dimension offset", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 3, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 2, 1}, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch([]tensor.Range{{From: 1, To: 4}, {From: 1, To: 3}, {From: 1, To: 2}}, x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
				{
					{0., 0.},
					{0., 0.},
					{0., 0.},
				},
				{
					{0., 0.},
					{0., 4.},
					{0., 4.},
				},
				{
					{0., 0.},
					{0., 4.},
					{0., 4.},
				},
				{
					{0., 0.},
					{0., 4.},
					{0., 4.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([[1,2],[3,4]]) tensor | Patch(nil, self) | returns same tensor", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{1., 2.},
				{3., 4.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Patch(nil, x)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{1., 2.},
				{3., 4.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Patch([3,4] tensor) then Device() | returns the device the inputs were created on", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch(nil, x2)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("two untracked [3,4] tensors | Patch | y is not gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch(nil, x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor and untracked [3,4] tensor | Patch | y is gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch(nil, x2)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		t.Run("untracked [3,4] tensor and grad-tracked [3,4] tensor | Patch | y is gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch(nil, x2)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("two grad-tracked Full(nil) tensors | Patch(nil) | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch(nil, x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked Full(nil) tensors | Patch(nil) | Gradient() returns nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 7., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch(nil, x2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("two untracked Full(nil) tensors | Patch(nil) then BackPropagate | gradient is nil", func(t *testing.T) {
			x1, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch(nil, x2)
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

		t.Run("grad-tracked [4,5] x1 and grad-tracked [3,3] x2 | Patch([1:4],[1:4]) then BackPropagate | gradient of x1 is 1 outside patch window, 0 inside; gradient of x2 is all-ones", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
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
			x2, err := tensor.Of([][]float64{
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

			y, err := x1.Patch([]tensor.Range{{From: 1, To: 4}, {From: 1, To: 4}}, x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Of([][]float64{
				{1., 1., 1., 1., 1.},
				{1., 0., 0., 0., 1.},
				{1., 0., 0., 0., 1.},
				{1., 0., 0., 0., 1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Of([][]float64{
				{1., 1., 1.},
				{1., 1., 1.},
				{1., 1., 1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [4,4,4] base and [2,2,2] patch | Patch([1:3],[1:3],[1:3]) then BackPropagate | base gradient is 1 outside window 0 inside, patch gradient is all-ones", func(t *testing.T) {
			x1, err := tensor.Full([]int{4, 4, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch([]tensor.Range{{From: 1, To: 3}, {From: 1, To: 3}, {From: 1, To: 3}}, x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full([]int{2, 2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensors filled with 3 and 2 | Patch(nil) | returns Full([1,2^20], 2)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch(nil, x2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large [1,2^20] grad-tracked tensors filled with 3 and 2 | Patch(nil) then BackPropagate | gradient of x1 is Full([1,2^20], 0), gradient of x2 is Full([1,2^20], 1)", func(t *testing.T) {
			n := 1 << 20

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch(nil, x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g1 := x1.Gradient()
			g2 := x2.Gradient()

			h1, err := tensor.Full([]int{1, n}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			h2, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g1.Equals(h1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := g2.Equals(h2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensors filled with 3 and 2 | concurrent repeated Patch over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x1, err := tensor.Full([]int{1, n}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1, n}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x1.Patch(nil, x2)
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := y.Equals(h); err != nil {
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

		t.Run("grad-tracked [3,4] x1 and untracked [3,4] x2 | Patch(nil) then ResetGradient(x1, false) | result is still gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch(nil, x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x1, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})

		t.Run("untracked [3,4] x1 and grad-tracked [3,4] x2 | Patch(nil) then ResetGradient(x2, false) | result is still gradient-tracked", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x1.Patch(nil, x2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x2, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("Full([2], 0.) | Patch([], nil) | returns error: source not on device", func(t *testing.T) {
			x, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Patch([]tensor.Range{}, nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Patch tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([2], 0.) and Full(nil, 0.) scalar | Patch([]) | returns error: dimension count mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{}, x2)
			if err == nil {
				t.Fatal("expected error because of incompatible number of dimensions")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: expected number of dimensions to match among source and target tensors: (0) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1,1], 0.) and Full([1,2], 0.) | Patch([]) | returns error: source exceeds target at dim 1", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{}, x2)
			if err == nil {
				t.Fatal("expected error because of exceeding patch size at dimension (1)")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: expected source tensor size not to exceed that of target tensor at dimension (1): (2) > (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([3], 0.) and Full([2], 0.) | Patch([{2,4}]) | returns error: index out of target range", func(t *testing.T) {
			x1, err := tensor.Full([]int{3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{{From: 2, To: 4}}, x2)
			if err == nil {
				t.Fatal("expected error because of incompatible index with target tensor")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: index incompatible with target tensor: expected index to fall in range [0,3] at dimension (0): got [2,4)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full(nil, 0.) and Full(nil, 0.) scalars | Patch with 1 range | returns error: index length exceeds dimensions", func(t *testing.T) {
			x1, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{{}}, x2)
			if err == nil {
				t.Fatal("expected error because of incompatible index len (1) with dimension len (0)")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: index incompatible with target tensor: expected index length to be smaller than or equal to the number of dimensions: (1) > (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1], 0.) and Full([1], 0.) | Patch with 2 ranges | returns error: index length exceeds dimensions", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{{}, {}}, x2)
			if err == nil {
				t.Fatal("expected error because of incompatible index len (2) with dimension len (1)")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: index incompatible with target tensor: expected index length to be smaller than or equal to the number of dimensions: (2) > (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1], 0.) and Full([1], 0.) | Patch([{1,1}]) | returns error: from not smaller than to", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{{From: 1, To: 1}}, x2)
			if err == nil {
				t.Fatal("expected error because of to index (1) not being larger than from index (1)")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: index incompatible with target tensor: expected range 'From' to be smaller than 'To' except for special both (0) case (fetchAll): (1) >= (1) at dimension (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1], 0.) and Full([1], 0.) | Patch([{-1,0}]) | returns error: negative from index at dimension 0", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{{From: -1, To: 0}}, x2)
			if err == nil {
				t.Fatal("expected error because of negative from index (-1)")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: index incompatible with target tensor: expected index to be in range [0,1) at dimension (0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1], 0.) and Full([1], 0.) | Patch([{1,2}]) | returns error: from index 1 out of range [0,1)", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{{From: 1, To: 2}}, x2)
			if err == nil {
				t.Fatal("expected error because of from index (1) being out of range [0,1) at dimension (0)")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: index incompatible with target tensor: expected index to be in range [0,1) at dimension (0): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1], 0.) and Full([1], 0.) | Patch([{0,2}]) | returns error: to index 2 out of range [0,1]", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{{From: 0, To: 2}}, x2)
			if err == nil {
				t.Fatal("expected error because of to index (2) being out of range [0,1) at dimension (0)")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: index incompatible with target tensor: expected index to fall in range [0,1] at dimension (0): got [0,2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([2], 0.) and Full([1], 0.) | Patch([{2,3}]) | returns error: from index 2 out of range [0,2)", func(t *testing.T) {
			x1, err := tensor.Full([]int{2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{{From: 2, To: 3}}, x2)
			if err == nil {
				t.Fatal("expected error because of from index (2) being out of range [0,2) at dimension (0)")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: index incompatible with target tensor: expected index to be in range [0,2) at dimension (0): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([2,2], 0.) and Full([2,1], 0.) | Patch([{0,2},{0,3}]) second range invalid | returns error: to index 3 out of range [0,2] at dimension 1", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{{From: 0, To: 2}, {From: 0, To: 3}}, x2)
			if err == nil {
				t.Fatal("expected error because of to index (3) being out of range [0,2) at dimension (1)")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: index incompatible with target tensor: expected index to fall in range [0,2] at dimension (1): got [0,3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([2,2,2], 0.) and Full([2,1,2], 0.) | Patch([{0,2},{3,4},{0,2}]) second range invalid | returns error: from index 3 out of range [0,2) at dimension 1", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{{From: 0, To: 2}, {From: 3, To: 4}, {From: 0, To: 2}}, x2)
			if err == nil {
				t.Fatal("expected error because of from index (3) being out of range [0,2) at dimension (1)")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: index incompatible with target tensor: expected index to be in range [0,2) at dimension (1): got (3)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([2,2,2], 0.) and Full([2,2,1], 0.) | Patch([{0,2},{0,2},{1,5}]) third range invalid | returns error: to index 5 out of range [0,2] at dimension 2", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 2, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 2, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{{From: 0, To: 2}, {From: 0, To: 2}, {From: 1, To: 5}}, x2)
			if err == nil {
				t.Fatal("expected error because of to index (5) being out of range [0,2) at dimension (2)")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: index incompatible with target tensor: expected index to fall in range [0,2] at dimension (2): got [1,5)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1,3], 0.) and Full([1,2], 0.) | Patch([{},{2,3}]) | returns error: index does not cover source at dim 1", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{{}, {From: 2, To: 3}}, x2)
			if err == nil {
				t.Fatal("expected error because of index not covering source tensor at dimension (1)")
			} else if err.Error() != "Patch input index or tensors' dimension validation failed: expected index to exactly cover source tensor at dimension (1): #[2,3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Full([2], 1.) on d1 and Full([1], 1.) on d2 | Patch([{0,1}]) | returns error: tensors not on the same device", func(t *testing.T) {
			x1, err := tensor.Full([]int{2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Patch([]tensor.Range{{From: 0, To: 1}}, x2)
			if err == nil {
				t.Fatal("expected error because of tensors not being on the same device")
			} else if err.Error() != fmt.Sprintf("Patch tensors' device validation failed: expected input tensor to be on %s", d1) {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
