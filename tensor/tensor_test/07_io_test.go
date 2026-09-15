package tensor_test

import (
	"sync"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestTransfer(t *testing.T) {
	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor | Transfer(srcDev -> dstDev) | returns same scalar", func(t *testing.T) {
			x, err := tensor.Of(2., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(2., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor | Transfer(srcDev -> dstDev) | returns [3, -1, 4]", func(t *testing.T) {
			x, err := tensor.Of([]float64{3., -1., 4.}, &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{3., -1., 4.}, &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor | Transfer(srcDev -> dstDev) | returns [[-2, 5], [1, 0]]", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{-2., 5.},
				{1., 0.},
			}, &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{-2., 5.},
				{1., 0.},
			}, &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2x2x5 tensor | Transfer(srcDev -> dstDev) | target equals source", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1., 2., 3., 4., 5.},
					{6., 7., 8., 9., 10.},
				},
				{
					{11., 12., 13., 14., 15.},
					{16., 17., 18., 19., 20.},
				},
			}, &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
				{
					{1., 2., 3., 4., 5.},
					{6., 7., 8., 9., 10.},
				},
				{
					{11., 12., 13., 14., 15.},
					{16., 17., 18., 19., 20.},
				},
			}, &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("scalar tensor | Transfer(srcDev -> dstDev) | Device() returns the target device", func(t *testing.T) {
			x, err := tensor.Of(1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != d2 {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", d2, d)
			}
		})

		t.Run("untracked scalar tensor | Transfer(srcDev -> dstDev) | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Of(1., &tensor.Config{
				Device:    d1,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked scalar tensor | Transfer(srcDev -> dstDev) | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Of(1., &tensor.Config{
				Device:    d1,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [2] tensor | Transfer(srcDev -> dstDev) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{2}, 7., &tensor.Config{
				Device:    d1,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [2] tensor | Transfer(srcDev -> dstDev) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{2}, 7., &tensor.Config{
				Device:    d1,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [2,3] tensor | Transfer(srcDev -> dstDev) then BackPropagate | y has nil gradient", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 3}, 3., &tensor.Config{
				Device:    d1,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
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

		t.Run("grad-tracked scalar tensor | Transfer then BackPropagate | gradient is all-ones scalar on the source device", func(t *testing.T) {
			x, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    d1,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full(nil, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [4] tensor | Transfer then BackPropagate | gradient is all-ones [4] on the source device", func(t *testing.T) {
			x, err := tensor.Full([]int{4}, 3., &tensor.Config{
				Device:    d1,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full([]int{4}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [3,4] tensor | Transfer then BackPropagate | gradient is all-ones [3,4] on the source device", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 3., &tensor.Config{
				Device:    d1,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [2,3,4] tensor | Transfer then BackPropagate | gradient is all-ones [2,3,4] on the source device", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 3, 4}, 3., &tensor.Config{
				Device:    d1,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full([]int{2, 3, 4}, 1., &tensor.Config{Device: d1})
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

		t.Run("Full([2^20], 7) large tensor | Transfer(srcDev -> dstDev) | returns Full([2^20], 7) on the target device", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([2^20], 7) large grad-tracked tensor | Transfer then BackPropagate | gradient is Full([2^20], 1) on the source device", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{n}, 7., &tensor.Config{
				Device:    d1,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full([]int{n}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([2^10], 7) tensor | concurrent repeated Transfer(srcDev -> dstDev) over every iteration | returns Full([2^10], 7) on the target device", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := tensor.Transfer(x, d2)
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

		t.Run("grad-tracked scalar tensor | Transfer(srcDev -> dstDev) then ResetGradient(source, false) | y stays gradient-tracked", func(t *testing.T) {
			x, err := tensor.Of(1., &tensor.Config{
				Device:    d1,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, d2)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})
	})

	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("2D tensor | Transfer(dev -> dev) | returns the same tensor instance", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{-2., 5.},
				{1., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Transfer(x, dev)
			if err != nil {
				t.Fatal(err)
			}

			if y != x {
				t.Fatal("expected Transfer to return the same tensor instance")
			}
		})

		// ============================== validations ==============================

		t.Run("nil tensor | Transfer | returns error: unsupported tensor implementation", func(t *testing.T) {
			_, err := tensor.Transfer(nil, dev)
			if err == nil {
				t.Fatal("expected error because of nil tensor input")
			} else if err.Error() != "Transfer tensor implementation validation failed: unsupported tensor implementation" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("valid tensor | Transfer to invalid device | returns error: invalid input device", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Transfer(x, tensor.Device(0))
			if err == nil {
				t.Fatal("expected error because of invalid target device")
			} else if err.Error() != "Transfer target device validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
