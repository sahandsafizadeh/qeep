package tensor_test

import (
	"fmt"
	"sync"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

/*
	Some initializers, accessors and operators are tested together rather than in isolation
	to resolve a chicken-and-egg dependency: verifying that a tensor was initialized correctly
	requires reading its values back, and trusting that a read is correct requires knowing the
	tensor was created correctly. By cross-validating both in the same suite, they establish a
	mutually consistent baseline that all other tests in this package build upon.
	Full, Of, At, Device, GradientTracked, and Equals are the primary functions under test here,
	as they underpin this baseline and are relied upon throughout the rest of the test suite.
*/

func Test_Full_At_Device_GradientTracked_ResetGradient_BackPropagate_Gradient(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("Full(nil, -1) scalar tensor | At() with no indices | returns -1", func(t *testing.T) {
			x, err := tensor.Full(nil, -1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := x.At(); err != nil {
				t.Fatal(err)
			} else if int(val) != -1 {
				t.Fatalf("expected (-1) as scalar tensor value, got (%f)", val)
			}
		})

		t.Run("Full([1], 9) 1D tensor | At(0) | returns 9", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := x.At(0); err != nil {
				t.Fatal(err)
			} else if int(val) != 9 {
				t.Fatalf("expected (9) as tensor value in position [0], got (%f)", val)
			}
		})

		t.Run("Full([1,1,1], 7) 3D tensor | At(0,0,0) | returns 7", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 1, 1}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := x.At(0, 0, 0); err != nil {
				t.Fatal(err)
			} else if int(val) != 7 {
				t.Fatalf("expected (7) as tensor value in position [0,0,0], got (%f)", val)
			}
		})

		t.Run("Full([1,2], 0) 2D tensor | At(i,j) for all positions | returns 0", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 2}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := x.At(0, 0); err != nil {
				t.Fatal(err)
			} else if int(val) != 0 {
				t.Fatalf("expected (0) as tensor value in position [0,0], got (%f)", val)
			}

			if val, err := x.At(0, 1); err != nil {
				t.Fatal(err)
			} else if int(val) != 0 {
				t.Fatalf("expected (0) as tensor value in position [0,1], got (%f)", val)
			}
		})

		t.Run("Full([3,1], -5) 2D tensor | At(i,j) for all positions | returns -5", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 1}, -5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := x.At(0, 0); err != nil {
				t.Fatal(err)
			} else if int(val) != -5 {
				t.Fatalf("expected (-5) as tensor value in position [0,0], got (%f)", val)
			}

			if val, err := x.At(1, 0); err != nil {
				t.Fatal(err)
			} else if int(val) != -5 {
				t.Fatalf("expected (-5) as tensor value in position [1,0], got (%f)", val)
			}

			if val, err := x.At(2, 0); err != nil {
				t.Fatal(err)
			} else if int(val) != -5 {
				t.Fatalf("expected (-5) as tensor value in position [2,0], got (%f)", val)
			}
		})

		t.Run("Full([4,3,2,1], 5) 4D tensor | At(i,j,k,u) for all positions | returns 5", func(t *testing.T) {
			x, err := tensor.Full([]int{4, 3, 2, 1}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			for i := range 4 {
				for j := range 3 {
					for k := range 2 {
						for u := range 1 {
							if val, err := x.At(i, j, k, u); err != nil {
								t.Fatal(err)
							} else if int(val) != 5 {
								t.Fatalf("expected (5) as tensor value in position [%d,%d,%d,%d], got (%f)", i, j, k, u, val)
							}
						}
					}
				}
			}
		})

		t.Run("Full(nil, 0) scalar tensor | Device() | returns the device it was created on", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if d := x.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("Full(nil, 0) with GradTrack true | GradientTracked() | returns true", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			if !x.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		t.Run("Full(nil, 0) with GradTrack false | GradientTracked() | returns false", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if x.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		t.Run("Full(nil, 0) with nil config | Device() and GradientTracked() | returns CPU and false", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., nil)
			if err != nil {
				t.Fatal(err)
			}

			if d := x.Device(); d != tensor.CPU {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", tensor.CPU, d)
			}
			if x.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		// =============== gradients ===============

		t.Run("Full(nil, 0) with GradTrack true | Gradient() | returns nil", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			if x.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("Full(nil, 0) with GradTrack false | Gradient() | returns nil", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if x.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("Full(nil, 3) scalar tensor with GradTrack false | BackPropagate | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(x)
			if err != nil {
				t.Fatal(err)
			}

			if x.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("Full(nil, 3) scalar tensor with GradTrack true | BackPropagate | Gradient().At() returns 1", func(t *testing.T) {
			x, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(x)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			if val, err := g.At(); err != nil {
				t.Fatal(err)
			} else if int(val) != 1 {
				t.Fatalf("expected (1) as gradient scalar value, got (%f)", val)
			}
		})

		t.Run("Full([3], 3) 1D tensor with GradTrack true | BackPropagate | Gradient().At(i) for all positions returns 1", func(t *testing.T) {
			x, err := tensor.Full([]int{3}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(x)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			for i := range 3 {
				if val, err := g.At(i); err != nil {
					t.Fatal(err)
				} else if int(val) != 1 {
					t.Fatalf("expected (1) as gradient value in position [%d], got (%f)", i, val)
				}
			}
		})

		t.Run("Full([2,2], 3) 2D tensor with GradTrack true | BackPropagate | Gradient().At(i,j) for all positions returns 1", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 2}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(x)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			for i := range 2 {
				for j := range 2 {
					if val, err := g.At(i, j); err != nil {
						t.Fatal(err)
					} else if int(val) != 1 {
						t.Fatalf("expected (1) as gradient value in position [%d,%d], got (%f)", i, j, val)
					}
				}
			}
		})

		t.Run("Full([1,2,3], 3) 3D tensor with GradTrack true | BackPropagate | Gradient().At(i,j,k) for all positions returns 1", func(t *testing.T) {
			x, err := tensor.Full([]int{1, 2, 3}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(x)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			for i := range 1 {
				for j := range 2 {
					for k := range 3 {
						if val, err := g.At(i, j, k); err != nil {
							t.Fatal(err)
						} else if int(val) != 1 {
							t.Fatalf("expected (1) as gradient value in position [%d,%d,%d], got (%f)", i, j, k, val)
						}
					}
				}
			}
		})

		t.Run("Full(nil, 0) with GradTrack true | ResetGradient(x, true) | GradientTracked() returns true", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.ResetGradient(x, true)
			if err != nil {
				t.Fatal(err)
			}

			if !x.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		t.Run("Full(nil, 0) with GradTrack true | ResetGradient(x, false) | GradientTracked() returns false", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if x.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		t.Run("Full(nil, 0) with GradTrack true | BackPropagate then ResetGradient(x, true) | Gradient() returns nil and GradientTracked() returns true", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(x)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.ResetGradient(x, true)
			if err != nil {
				t.Fatal(err)
			}

			if x.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
			if !x.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		t.Run("Full(nil, 0) with GradTrack true | BackPropagate then ResetGradient(x, false) | Gradient() returns nil and GradientTracked() returns false", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(x)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if x.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
			if x.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})
	})
}

func Test_Of_At_Device_GradientTracked_ResetGradient_BackPropagate_Gradient(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("Of(2) scalar | At() | returns 2", func(t *testing.T) {
			x, err := tensor.Of(2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := x.At(); err != nil {
				t.Fatal(err)
			} else if int(val) != 2 {
				t.Fatalf("expected (2) as scalar tensor value, got (%f)", val)
			}
		})

		t.Run("Of([3]) 1D tensor | At(0) | returns 3", func(t *testing.T) {
			x, err := tensor.Of([]float64{3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := x.At(0); err != nil {
				t.Fatal(err)
			} else if int(val) != 3 {
				t.Fatalf("expected (3) as tensor value in position [0], got (%f)", val)
			}
		})

		t.Run("Of([1, 4]) 1D tensor | At(i) for all positions | returns 1 and 4", func(t *testing.T) {
			x, err := tensor.Of([]float64{1., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := x.At(0); err != nil {
				t.Fatal(err)
			} else if int(val) != 1 {
				t.Fatalf("expected (1) as tensor value in position [0], got (%f)", val)
			}

			if val, err := x.At(1); err != nil {
				t.Fatal(err)
			} else if int(val) != 4 {
				t.Fatalf("expected (4) as tensor value in position [1], got (%f)", val)
			}
		})

		t.Run("Of([[-1], [-2]]) 2D tensor | At(i,j) for all positions | returns -1 and -2", func(t *testing.T) {
			x, err := tensor.Of([][]float64{{-1.}, {-2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if val, err := x.At(0, 0); err != nil {
				t.Fatal(err)
			} else if int(val) != -1 {
				t.Fatalf("expected (-1) as tensor value in position [0,0], got (%f)", val)
			}

			if val, err := x.At(1, 0); err != nil {
				t.Fatal(err)
			} else if int(val) != -2 {
				t.Fatalf("expected (-2) as tensor value in position [1,0], got (%f)", val)
			}
		})

		t.Run("Of(3x3x3 tensor) | At(i,j,k) for all positions | returns expected values", func(t *testing.T) {
			data := [][][]float64{
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

			x, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			for i := range data {
				for j := range data[0] {
					for k := range data[0][0] {
						if val, err := x.At(i, j, k); err != nil {
							t.Fatal(err)
						} else if int(val) != int(data[i][j][k]) {
							t.Fatalf("expected (%f) as tensor value in position [%d,%d,%d], got (%f)",
								data[i][j][k], i, j, k, val)
						}
					}
				}
			}
		})

		t.Run("Of(1x2x3x4 tensor) | At(i,j,k,u) for all positions | returns 1 through 4 per row", func(t *testing.T) {
			data := [][][][]float64{
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

			x, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			for i := range data {
				for j := range data[0] {
					for k := range data[0][0] {
						for u := range data[0][0][0] {
							if val, err := x.At(i, j, k, u); err != nil {
								t.Fatal(err)
							} else if int(val) != int(data[i][j][k][u]) {
								t.Fatalf("expected (%f) as tensor value in position [%d,%d,%d,%d], got (%f)",
									data, i, j, k, u, val)
							}
						}
					}
				}
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("Of([2^20]) large 1D tensor | At(i) for all positions | matches source data", func(t *testing.T) {
			n := 1 << 20

			data := make([]float64, n)
			for i := range data {
				data[i] = float64(i)
			}

			x, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			for i := range n {
				if val, err := x.At(i); err != nil {
					t.Fatal(err)
				} else if val != data[i] {
					t.Fatalf("expected (%f) as tensor value in position [%d], got (%f)", data[i], i, val)
				}
			}
		})

		t.Run("Of([2^10]) 1D tensor | concurrent repeated Of then At(i) over every position | matches source data", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			data := make([]float64, n)
			for i := range data {
				data[i] = float64(i)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						x, err := tensor.Of(data, &tensor.Config{Device: dev})
						if err != nil {
							t.Error(err)
							return
						}

						for i := range n {
							if val, err := x.At(i); err != nil {
								t.Error(err)
								return
							} else if val != data[i] {
								t.Errorf("expected (%f) as tensor value in position [%d], got (%f)", data[i], i, val)
								return
							}
						}
					}
				})
			}
			wg.Wait()
		})

		t.Run("Of([2^10]) large 1D tensor | concurrent At(i) over every position | matches source data", func(t *testing.T) {
			const (
				n  = 1 << 10
				ng = 1 << 8
			)

			data := make([]float64, n)
			for i := range data {
				data[i] = float64(i)
			}

			x, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for i := range n {
						if val, err := x.At(i); err != nil {
							t.Error(err)
							return
						} else if val != data[i] {
							t.Errorf("expected (%f) as tensor value in position [%d], got (%f)", data[i], i, val)
							return
						}
					}
				})
			}
			wg.Wait()
		})

		// ============================== side effects ==============================

		t.Run("Of([5]) 1D tensor from a caller-owned slice | At(0) after mutating source | returns 5", func(t *testing.T) {
			data := []float64{5.}

			ten, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			data[0] = 3.

			if val, err := ten.At(0); err != nil {
				t.Fatal(err)
			} else if int(val) != 5 {
				t.Fatalf("expected (5) as tensor value in position [0], got (%f)", val)
			}
		})

		t.Run("Of([[[[5]]]]) 4D tensor from a caller-owned slice | At(0,0,0,0) after mutating source | returns 5", func(t *testing.T) {
			data := [][][][]float64{{{{5.}}}}

			ten, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			data[0][0][0][0] = 3.

			if val, err := ten.At(0, 0, 0, 0); err != nil {
				t.Fatal(err)
			} else if int(val) != 5 {
				t.Fatalf("expected (5) as tensor value in position [0,0,0,0], got (%f)", val)
			}
		})

		// ============================== validations ==============================

		t.Run("Of([]float64{}) | returns error: zero length along dimension", func(t *testing.T) {
			_, err := tensor.Of([]float64{}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of zero len along dimension (0)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: expected data to not have zero length along any dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][]float64{}) | returns error: zero length along dimension", func(t *testing.T) {
			_, err := tensor.Of([][]float64{}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of zero len along dimension (0)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: expected data to not have zero length along any dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][][]float64{}) | returns error: zero length along dimension", func(t *testing.T) {
			_, err := tensor.Of([][][]float64{}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of zero len along dimension (0)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: expected data to not have zero length along any dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][][][]float64{}) | returns error: zero length along dimension", func(t *testing.T) {
			_, err := tensor.Of([][][][]float64{}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of zero len along dimension (0)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: expected data to not have zero length along any dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][]float64{{}, {}}) | returns error: zero length along inner dimension", func(t *testing.T) {
			_, err := tensor.Of([][]float64{{}, {}}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of zero len along dimension (1)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: expected data to not have zero length along any dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][]float64{{}, {-1}}) | returns error: zero length along inner dimension", func(t *testing.T) {
			_, err := tensor.Of([][]float64{{}, {-1.}}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of zero len along dimension (1)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: expected data to not have zero length along any dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of([][][]float64{{{}}}) | returns error: zero length along inner dimension", func(t *testing.T) {
			_, err := tensor.Of([][][]float64{{{}}}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of zero len along dimension (1)")
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: expected data to not have zero length along any dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of(3x3x3 with inconsistent last row) | returns error: unequal lengths along dimension", func(t *testing.T) {
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
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: expected data to have have equal length along every dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of(1x3x3x3 with inconsistent inner sub-tensor) | returns error: unequal lengths along dimension", func(t *testing.T) {
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
			} else if err.Error() != fmt.Sprintf("%s initialization: Of input data validation failed: expected data to have have equal length along every dimension", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Of(2x3x1x3 with inconsistent outer batch) | returns error: unequal lengths along dimension", func(t *testing.T) {
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

		t.Run("Of([1]) with invalid device | returns error: invalid device", func(t *testing.T) {
			_, err := tensor.Of([]float64{1}, &tensor.Config{Device: -1})
			if err == nil {
				t.Fatal("expected error because of invalid input device")
			} else if err.Error() != "Of tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestDevice(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("Full(nil, 0) scalar tensor | Device() | returns the device it was created on", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if d := x.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("Of(0) scalar tensor | Device() | returns the device it was created on", func(t *testing.T) {
			x, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if d := x.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("Full(nil, 0) with nil config | Device() | returns CPU", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., nil)
			if err != nil {
				t.Fatal(err)
			}

			if d := x.Device(); d != tensor.CPU {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", tensor.CPU, d)
			}
		})

		t.Run("Of(0) with nil config | Device() | returns CPU", func(t *testing.T) {
			x, err := tensor.Of(0., nil)
			if err != nil {
				t.Fatal(err)
			}

			if d := x.Device(); d != tensor.CPU {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", tensor.CPU, d)
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("Full(nil, 0) scalar tensor | concurrent repeated Device() over every iteration | always returns the creation device", func(t *testing.T) {
			const (
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						if d := x.Device(); d != dev {
							t.Errorf("expected tensor's device to be (%s), got (%s)", dev, d)
							return
						}
					}
				})
			}
			wg.Wait()
		})

		// ============================== side effects ==============================

		t.Run("Full(nil, 0) built from a Config pointer | Device() after mutating config.Device | returns the original creation device", func(t *testing.T) {
			conf := &tensor.Config{Device: dev}

			x, err := tensor.Full(nil, 0., conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.Device++

			if d := x.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("Of(0) built from a Config pointer | Device() after mutating config.Device | returns the original creation device", func(t *testing.T) {
			conf := &tensor.Config{Device: dev}

			x, err := tensor.Of(0., conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.Device++

			if d := x.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})
	})
}

func TestGradientTracked(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("Full(nil, 0) with GradTrack true | GradientTracked() | returns true", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			if !x.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		t.Run("Of(0) with GradTrack true | GradientTracked() | returns true", func(t *testing.T) {
			x, err := tensor.Of(0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			if !x.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		t.Run("Full(nil, 0) with GradTrack false | GradientTracked() | returns false", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if x.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		t.Run("Of(0) with GradTrack false | GradientTracked() | returns false", func(t *testing.T) {
			x, err := tensor.Of(0., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if x.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		t.Run("Full(nil, 0) with nil config | GradientTracked() | returns false", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., nil)
			if err != nil {
				t.Fatal(err)
			}

			if x.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		t.Run("Of(0) with nil config | GradientTracked() | returns false", func(t *testing.T) {
			x, err := tensor.Of(0., nil)
			if err != nil {
				t.Fatal(err)
			}

			if x.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("Full(nil, 0) with GradTrack true | concurrent repeated GradientTracked() over every iteration | always returns true", func(t *testing.T) {
			const (
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						if !x.GradientTracked() {
							t.Error("expected tensor to be gradient tracked")
							return
						}
					}
				})
			}
			wg.Wait()
		})

		// ============================== side effects ==============================

		t.Run("Full(nil, 0) with GradTrack true from a Config pointer | GradientTracked() after setting config.GradTrack to false | returns true", func(t *testing.T) {
			conf := &tensor.Config{
				Device:    dev,
				GradTrack: true,
			}

			x, err := tensor.Full(nil, 0., conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.GradTrack = false

			if !x.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		t.Run("Of(0) with GradTrack true from a Config pointer | GradientTracked() after setting config.GradTrack to false | returns true", func(t *testing.T) {
			conf := &tensor.Config{
				Device:    dev,
				GradTrack: true,
			}

			x, err := tensor.Of(0., conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.GradTrack = false

			if !x.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})
	})
}

func TestEquals(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("Full(nil, 3) scalar == Full(nil, 3) scalar | Equals() | returns true", func(t *testing.T) {
			x1, err := tensor.Full(nil, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x1.Equals(x2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected equal scalar tensors to be equal")
			}
		})

		t.Run("Full(nil, 3) scalar != Full(nil, 4) scalar | Equals() | returns false", func(t *testing.T) {
			x1, err := tensor.Full(nil, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full(nil, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x1.Equals(x2); err != nil {
				t.Fatal(err)
			} else if eq {
				t.Fatal("expected scalar tensors with different values to not be equal")
			}
		})

		t.Run("Full(nil, 3) gradtrack == Full(nil, 3) non-gradtrack | Equals() | returns true", func(t *testing.T) {
			x1, err := tensor.Full(nil, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
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

			if eq, err := x1.Equals(x2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected scalar tensors with equal values to be equal regardless of gradtrack config")
			}
		})

		t.Run("Of([1,2,3]) == Of([1,2,3]) 1D tensors | Equals() | returns true", func(t *testing.T) {
			x1, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x1.Equals(x2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected equal 1D tensors to be equal")
			}
		})

		t.Run("Of([1,2,3]) != Of([1,2,4]) 1D tensors | Equals() | returns false", func(t *testing.T) {
			x1, err := tensor.Of([]float64{1., 2., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{1., 2., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x1.Equals(x2); err != nil {
				t.Fatal(err)
			} else if eq {
				t.Fatal("expected 1D tensors with a differing element to not be equal")
			}
		})

		t.Run("Of(2x3 matrix) == Of(2x3 same matrix) | Equals() | returns true", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x1.Equals(x2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected equal 2D tensors to be equal")
			}
		})

		t.Run("Of(2x3 matrix) != Of(2x3 different matrix) | Equals() | returns false", func(t *testing.T) {
			x1, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x1.Equals(x2); err != nil {
				t.Fatal(err)
			} else if eq {
				t.Fatal("expected 2D tensors with a differing element to not be equal")
			}
		})

		t.Run("Full([2,3,4,2], 7) 4D tensor | Equals(itself) | returns true", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 3, 4, 2}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x.Equals(x); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensor to equal itself")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("Of([2^20]) large 1D tensors differing in one element | Equals() | returns false", func(t *testing.T) {
			n := 1 << 20

			data := make([]float64, n)
			for i := range data {
				data[i] = float64(i)
			}

			x1, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			data[n-1]--

			x2, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := x1.Equals(x2); err != nil {
				t.Fatal(err)
			} else if eq {
				t.Fatal("expected large 1D tensors differing in a single element to not be equal")
			}
		})

		t.Run("equal Of([2^10]) 1D tensors | concurrent repeated Equals() | never errors", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			data := make([]float64, n)
			for i := range data {
				data[i] = float64(i)
			}

			x1, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						if eq, err := x1.Equals(x2); err != nil {
							t.Error(err)
							return
						} else if !eq {
							t.Error("expected equal 1D tensors to be equal")
							return
						}
					}
				})
			}
			wg.Wait()
		})

		// ============================== validations ==============================

		t.Run("Full(nil) scalar | Equals(nil) | returns error: nil tensor fails device validation", func(t *testing.T) {
			x, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Equals(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != fmt.Sprintf("Equals tensors' device validation failed: expected input tensor to be on %s", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full(nil) scalar | Equals(Full([1])) | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Equals(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Equals tensors' dimension validation failed: expected number of dimensions to match: (0) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1]) | Equals(Full([1,1])) | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{1, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Equals(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Equals tensors' dimension validation failed: expected number of dimensions to match: (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([6,5,2]) | Equals(Full([6,5])) | returns error: number of dimensions mismatch", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 5}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Equals(x2)
			if err == nil {
				t.Fatal("expected error because of tensors having different number of dimensions")
			} else if err.Error() != "Equals tensors' dimension validation failed: expected number of dimensions to match: (3) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1,2]) | Equals(Full([2,1])) | returns error: size mismatch at dimension 0", func(t *testing.T) {
			x1, err := tensor.Full([]int{1, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Equals(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (0)")
			} else if err.Error() != "Equals tensors' dimension validation failed: expected sizes to match at dimension (0): (1) != (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([6,5,2]) | Equals(Full([6,4,2])) | returns error: size mismatch at dimension 1", func(t *testing.T) {
			x1, err := tensor.Full([]int{6, 5, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{6, 4, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x1.Equals(x2)
			if err == nil {
				t.Fatal("expected error because of incompatible sizes at dimension (1)")
			} else if err.Error() != "Equals tensors' dimension validation failed: expected sizes to match at dimension (1): (5) != (4)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
