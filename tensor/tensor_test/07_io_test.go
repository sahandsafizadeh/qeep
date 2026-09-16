package tensor_test

import (
	"archive/zip"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
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

func TestSaveLoad(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor | Save then Load | returns same scalar", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{Device: dev})
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

		t.Run("1D tensor | Save then Load | returns [3, -1, 4]", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of([]float64{3., -1., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{3., -1., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor | Save then Load | returns [[-2, 5], [1, 0]]", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of([][]float64{
				{-2., 5.},
				{1., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{-2., 5.},
				{1., 0.},
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

		t.Run("2x2x5 tensor | Save then Load | loaded tensor equals source", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of([][][]float64{
				{
					{1., 2., 3., 4., 5.},
					{6., 7., 8., 9., 10.},
				},
				{
					{11., 12., 13., 14., 15.},
					{16., 17., 18., 19., 20.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{Device: dev})
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

		t.Run("1x2x3x4 tensor | Save then Load | loaded tensor equals source", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of([][][][]float64{
				{
					{
						{1., 2., 3., 4.},
						{5., 6., 7., 8.},
						{9., 10., 11., 12.},
					},
					{
						{13., 14., 15., 16.},
						{17., 18., 19., 20.},
						{21., 22., 23., 24.},
					},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][][]float64{
				{
					{
						{1., 2., 3., 4.},
						{5., 6., 7., 8.},
						{9., 10., 11., 12.},
					},
					{
						{13., 14., 15., 16.},
						{17., 18., 19., 20.},
						{21., 22., 23., 24.},
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

		t.Run("scalar tensor | Save then Load | Device() returns the load device", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("scalar tensor | Save then Load with GradTrack true | GradientTracked() returns true", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of(2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		t.Run("scalar tensor | Save then Load with GradTrack false | GradientTracked() returns false", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of(2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		t.Run("scalar tensor | Save then Load with nil config | Device() and GradientTracked() return CPU and false", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, nil)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != tensor.CPU {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", tensor.CPU, d)
			}
			if y.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		// =============== gradients ===============

		t.Run("scalar tensor | Save then Load with GradTrack true | Gradient() returns nil", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("scalar tensor | Save then Load with GradTrack false | Gradient() returns nil", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("scalar tensor | Save then Load with GradTrack false then BackPropagate | Gradient() returns nil", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
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

		t.Run("scalar tensor | Save then Load with GradTrack true then BackPropagate | Gradient() Equals Full(nil, 1.)", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := y.Gradient()

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

		t.Run("[3] tensor | Save then Load with GradTrack true then BackPropagate | Gradient() Equals Full([3], 1.)", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of([]float64{3., -1., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := y.Gradient()

			h, err := tensor.Full([]int{3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[2,2] tensor | Save then Load with GradTrack true then BackPropagate | Gradient() Equals Full([2,2], 1.)", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of([][]float64{
				{-2., 5.},
				{1., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := y.Gradient()

			h, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[2,2,5] tensor | Save then Load with GradTrack true then BackPropagate | Gradient() Equals Full([2,2,5], 1.)", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of([][][]float64{
				{
					{1., 2., 3., 4., 5.},
					{6., 7., 8., 9., 10.},
				},
				{
					{11., 12., 13., 14., 15.},
					{16., 17., 18., 19., 20.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := y.Gradient()

			h, err := tensor.Full([]int{2, 2, 5}, 1., &tensor.Config{Device: dev})
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

		t.Run("Full([2^20], 7) large tensor | Save then Load | returns Full([2^20], 7)", func(t *testing.T) {
			n := 1 << 20

			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== side effects ==============================

		t.Run("valid archive | Load then mutating the config | Device() returns the load device", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of([]float64{1., 2.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			conf := &tensor.Config{Device: dev}

			y, err := tensor.Load(path, conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.Device++

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("valid archive | Load then mutating the config | GradientTracked() returns the load setting", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of([]float64{1., 2.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			conf := &tensor.Config{
				Device:    dev,
				GradTrack: true,
			}

			y, err := tensor.Load(path, conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.GradTrack = false

			if !y.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("nil tensor | Save | returns error: unsupported tensor implementation", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			err := tensor.Save(nil, path)
			if err == nil {
				t.Fatal("expected error because of nil tensor input")
			} else if err.Error() != "Save tensor implementation validation failed: unsupported tensor implementation" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("valid tensor | Save to unwritable path | returns error: failed to create tensor file", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 2.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			path := filepath.Join(t.TempDir(), "missing-dir", "tensor.qeep")

			err = tensor.Save(x, path)
			if err == nil {
				t.Fatal("expected error because of unwritable path")
			} else if !strings.HasPrefix(err.Error(), "Save operation failed: failed to create tensor file") {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("valid archive | Load with invalid device config | returns error: invalid input device", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of([]float64{1., 2.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Load(path, &tensor.Config{Device: -1})
			if err == nil {
				t.Fatal("expected error because of invalid device config")
			} else if err.Error() != "Load tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("non-existing path | Load | returns error: failed to open tensor file", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "does-not-exist.qeep")

			_, err := tensor.Load(path, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-existing path")
			} else if !strings.HasPrefix(err.Error(), "Load operation failed: failed to open tensor file") {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("non-archive file | Load | returns error: failed to open tensor archive", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			err := os.WriteFile(path, []byte("this is not a zip archive"), 0o644)
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Load(path, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because path is not an archive")
			} else if !strings.HasPrefix(err.Error(), "Load operation failed: failed to open tensor archive") {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("archive missing expected files | Load | returns error: failed to open \"meta\" file", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			err := createTensorArchive(path, map[string][]byte{
				"unexpected": {1, 2, 3},
			})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Load(path, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because archive is missing expected files")
			} else if err.Error() != "Load operation failed: failed to open \"meta\" file of tensor archive: open meta: file does not exist" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("archive missing expected files | Load | returns error: failed to open \"data\" file", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			err := createTensorArchive(path, map[string][]byte{
				"meta": int64bytes([]int64{1, 2, 3}),
			})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Load(path, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because archive is missing expected files")
			} else if err.Error() != "Load operation failed: failed to open \"data\" file of tensor archive: open data: file does not exist" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("archive with invalid meta content type | Load | returns error: corrupt \"meta\" file", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			err := createTensorArchive(path, map[string][]byte{
				"meta": []byte("dims"),
				"data": float64bytes([]float64{1., 2.}),
			})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Load(path, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because meta content is not a valid int64 encoding")
			} else if err.Error() != "Load operation failed: corrupt \"meta\" file of tensor archive: size (4) is not a multiple of (8)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("archive with unreadable meta content | Load | returns error: invalid dimension size", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			err := createTensorArchive(path, map[string][]byte{
				"meta": int64bytes([]int64{0}),
				"data": float64bytes([]float64{1., 2.}),
			})

			_, err = tensor.Load(path, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because meta content is not a valid dimension")
			} else if err.Error() != "Load operation failed: corrupt tensor archive: invalid dimension size (0) at position (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("archive with unreadable data content | Load | returns error: corrupt \"data\" file", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			err := createTensorArchive(path, map[string][]byte{
				"meta": int64bytes([]int64{2}),
				"data": {1, 2, 3, 4},
			})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Load(path, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because data content can't be read into a data slice")
			} else if err.Error() != "Load operation failed: corrupt \"data\" file of tensor archive: size (4) is not a multiple of (8)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("archive with dims not matching data | Load | returns error: dimensions do not match the number of elements", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			err := createTensorArchive(path, map[string][]byte{
				"meta": int64bytes([]int64{3}),
				"data": float64bytes([]float64{1., 2.}),
			})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Load(path, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because dims don't match the data")
			} else if err.Error() != "Load operation failed: corrupt tensor archive: dimensions [3] do not match the number of elements (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor | Save(srcDev) then Load(dstDev) | returns same scalar on dstDev", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of(2., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{Device: d2})
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

		t.Run("1D tensor | Save(srcDev) then Load(dstDev) | returns [3, -1, 4] on dstDev", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of([]float64{3., -1., 4.}, &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{Device: d2})
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

		t.Run("2D tensor | Save(srcDev) then Load(dstDev) | returns [[-2, 5], [1, 0]] on dstDev", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

			x, err := tensor.Of([][]float64{
				{-2., 5.},
				{1., 0.},
			}, &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{Device: d2})
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

		t.Run("2x2x5 tensor | Save(srcDev) then Load(dstDev) | loaded tensor equals source on dstDev", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensor.qeep")

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

			err = tensor.Save(x, path)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Load(path, &tensor.Config{Device: d2})
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
	})
}

/* ----- helpers ----- */

func createTensorArchive(path string, files map[string][]byte) (err error) {
	f, err := os.Create(path)
	if err != nil {
		return err
	}

	defer func() { err = f.Close() }()

	zw := zip.NewWriter(f)
	for name, content := range files {
		w, err := zw.Create(name)
		if err != nil {
			return err
		}

		_, err = w.Write(content)
		if err != nil {
			return err
		}
	}

	if err := zw.Close(); err != nil {
		return err
	}

	return nil
}

func int64bytes(arr []int64) []byte {
	n := len(arr) * 8
	res := make([]byte, n)

	for i, v := range arr {
		c := uint64(v)
		binary.LittleEndian.PutUint64(res[i*8:], c)
	}

	return res
}

func float64bytes(arr []float64) []byte {
	n := len(arr) * 8
	b := make([]byte, n)

	for i, v := range arr {
		c := math.Float64bits(v)
		binary.LittleEndian.PutUint64(b[i*8:], c)
	}

	return b
}
