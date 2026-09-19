package tensor_test

import (
	"math"
	"sync"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestSum(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor | Sum() | returns 9", func(t *testing.T) {
			x, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Sum(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [6, 4] | Sum() | returns 10", func(t *testing.T) {
			x, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Sum(), &tensor.Config{Device: dev})
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

		t.Run("3D tensor shape [3,1,4] | Sum() | returns 36", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{{9., -1., 8., 6.}},
				{{-5., 4., 1., 0.}},
				{{2., 8., 7., -3.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Sum(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(36., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 7 | Sum() | returns 7*2^20", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Sum(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(7.*float64(n), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor | concurrent repeated Sum() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(7.*float64(n), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := tensor.Of(x.Sum(), &tensor.Config{Device: dev})
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
	})
}

func TestMax(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor | Max() | returns 9", func(t *testing.T) {
			x, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Max(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [6, 4] | Max() | returns 6", func(t *testing.T) {
			x, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Max(), &tensor.Config{Device: dev})
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

		t.Run("3D tensor shape [3,1,4] | Max() | returns 9", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{{9., -1., 8., 6.}},
				{{-5., 4., 1., 0.}},
				{{2., 8., 7., -3.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Max(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 7 | Max() | returns 7", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Max(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor | concurrent repeated Max() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := tensor.Of(x.Max(), &tensor.Config{Device: dev})
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
	})
}

func TestMin(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor | Min() | returns 9", func(t *testing.T) {
			x, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Min(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [6, 4] | Min() | returns 4", func(t *testing.T) {
			x, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Min(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D tensor shape [3,1,4] | Min() | returns -5", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{{9., -1., 8., 6.}},
				{{-5., 4., 1., 0.}},
				{{2., 8., 7., -3.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Min(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(-5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 7 | Min() | returns 7", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Min(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor | concurrent repeated Min() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := tensor.Of(x.Min(), &tensor.Config{Device: dev})
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
	})
}

func TestAvg(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor | Avg() | returns 9", func(t *testing.T) {
			x, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Avg(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [6, 4] | Avg() | returns 5", func(t *testing.T) {
			x, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Avg(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D tensor shape [3,1,4] | Avg() | returns 3", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{{9., -1., 8., 6.}},
				{{-5., 4., 1., 0.}},
				{{2., 8., 7., -3.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Avg(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor shape [2,3] all-1e308 | Avg() | returns 1e308 despite overflow in naive sum", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{1e308, 1e308, 1e308},
				{1e308, 1e308, 1e308},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Avg(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(1e308, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 7 | Avg() | returns 7", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Avg(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor | concurrent repeated Avg() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := tensor.Of(x.Avg(), &tensor.Config{Device: dev})
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
	})
}

func TestVar(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor | Var() | returns 0", func(t *testing.T) {
			x, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Var(), &tensor.Config{Device: dev})
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

		t.Run("1D tensor [-sqrt2, sqrt2] | Var() | returns 4", func(t *testing.T) {
			x, err := tensor.Of([]float64{-math.Sqrt2, math.Sqrt2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Var(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D tensor shape [1,3,1] | Var() | returns 4", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{{{-2.}, {0.}, {2.}}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Var(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [5, 5, 5] | Var() | returns 0", func(t *testing.T) {
			x, err := tensor.Of([]float64{5., 5., 5.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Var(), &tensor.Config{Device: dev})
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

		t.Run("2D tensor shape [2,4] | Var() | returns 6", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{1., 2., 3., 4.},
				{5., 6., 7., 8.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Var(), &tensor.Config{Device: dev})
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

		t.Run("1D tensor [-3, 0, 3] | Var() | returns 9", func(t *testing.T) {
			x, err := tensor.Of([]float64{-3., 0., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Var(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D tensor shape [2,2,3] | Var() | returns 13", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := tensor.Of(x.Var(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(13., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor shape [3,1] with values offset around 1e8 | Var() | returns 9 without catastrophic cancellation", func(t *testing.T) {
			x, err := tensor.Of([][]float64{{1e8 + 3}, {1e8}, {1e8 - 3}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Var(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 7 | Var() | returns 0", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Var(), &tensor.Config{Device: dev})
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

		t.Run("[1,2^10] tensor | concurrent repeated Var() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := tensor.Of(x.Var(), &tensor.Config{Device: dev})
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
	})
}

func TestStd(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor | Std() | returns 0", func(t *testing.T) {
			x, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Std(), &tensor.Config{Device: dev})
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

		t.Run("1D tensor [-sqrt2, sqrt2] | Std() | returns 2", func(t *testing.T) {
			x, err := tensor.Of([]float64{-math.Sqrt2, math.Sqrt2}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Std(), &tensor.Config{Device: dev})
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

		t.Run("3D tensor shape [1,3,1] | Std() | returns 2", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{{{-2.}, {0.}, {2.}}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Std(), &tensor.Config{Device: dev})
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

		t.Run("1D tensor [7, 7, 7] | Std() | returns 0", func(t *testing.T) {
			x, err := tensor.Of([]float64{7., 7., 7.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Std(), &tensor.Config{Device: dev})
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

		t.Run("1D tensor [-3, 0, 3] | Std() | returns 3", func(t *testing.T) {
			x, err := tensor.Of([]float64{-3., 0., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Std(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor shape [2,4] | Std() | returns sqrt(6)", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{1., 2., 3., 4.},
				{5., 6., 7., 8.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Std(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(math.Sqrt(6), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D tensor shape [2,2,3] | Std() | returns sqrt(13)", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := tensor.Of(x.Std(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(math.Sqrt(13), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor shape [3,1] with values offset around 1e8 | Std() | returns 3 without catastrophic cancellation", func(t *testing.T) {
			x, err := tensor.Of([][]float64{{1e8 + 3}, {1e8}, {1e8 - 3}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Std(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 7 | Std() | returns 0", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Std(), &tensor.Config{Device: dev})
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

		t.Run("[1,2^10] tensor | concurrent repeated Std() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := tensor.Of(x.Std(), &tensor.Config{Device: dev})
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
	})
}

func TestMean(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("scalar tensor | Mean() | returns 9", func(t *testing.T) {
			x, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Mean(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [6, 4] | Mean() | returns 5", func(t *testing.T) {
			x, err := tensor.Of([]float64{6., 4.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Mean(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D tensor shape [3,1,4] | Mean() | returns 3", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{{9., -1., 8., 6.}},
				{{-5., 4., 1., 0.}},
				{{2., 8., 7., -3.}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Mean(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D tensor shape [2,3] all-1e308 | Mean() | returns 1e308 despite overflow in naive sum", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{1e308, 1e308, 1e308},
				{1e308, 1e308, 1e308},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Mean(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(1e308, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("large [1,2^20] tensor filled with 7 | Mean() | returns 7", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Of(x.Mean(), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor | concurrent repeated Mean() over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := tensor.Of(x.Mean(), &tensor.Config{Device: dev})
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
	})
}

func TestSumAlong(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("Full([1], 1) tensor | SumAlong(0) | returns scalar 1", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.SumAlong(0)
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

		t.Run("Full([3], 1) tensor | SumAlong(0) | returns scalar 3", func(t *testing.T) {
			x, err := tensor.Full([]int{3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,4,5], 1) tensor | SumAlong(0) | returns Full([4,5], 3)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4, 5}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{4, 5}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,4,5], 1) tensor | SumAlong(1) | returns Full([3,5], 4)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4, 5}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.SumAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3, 5}, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,4,5], 1) tensor | SumAlong(2) | returns Full([3,4], 5)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4, 5}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.SumAlong(2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3, 4}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("4D tensor shape [2,2,2,2] | SumAlong(1) | returns [2,2,2] tensor", func(t *testing.T) {
			x, err := tensor.Of([][][][]float64{
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

			y, err := x.SumAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
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

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("4D tensor shape [2,3,2,2] | SumAlong(3) | returns [2,3,2] tensor", func(t *testing.T) {
			x, err := tensor.Of([][][][]float64{
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

			y, err := x.SumAlong(3)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
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

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | SumAlong(0) then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.SumAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | SumAlong(0) | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | SumAlong(0) | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | SumAlong(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | SumAlong(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | SumAlong(0) then BackPropagate | y has nil gradient", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("grad-tracked [2,4,4] tensor | SumAlong(1) then BackPropagate | gradient of x is all-ones [2,4,4]", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 4, 4}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{2, 4, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [4,3] tensor | SumAlong(0) then BackPropagate | gradient of x is all-ones [4,3]", func(t *testing.T) {
			x, err := tensor.Full([]int{4, 3}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{4, 3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [2,3,4] tensor | SumAlong(2) then BackPropagate | gradient of x is all-ones [2,3,4]", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 3, 4}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{2, 3, 4}, 1., &tensor.Config{Device: dev})
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

		t.Run("large [1,2^20] tensor filled with 7 | SumAlong(1) | returns Full([1], 7*2^20)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.SumAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 7.*float64(n), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor | SumAlong(1) then BackPropagate | gradient of x is all-ones [1,2^20]", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
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

		t.Run("[1,2^10] tensor | concurrent repeated SumAlong(1) over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 7.*float64(n), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x.SumAlong(1)
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

		t.Run("grad-tracked [3,4] tensor | SumAlong(0) then ResetGradient(source, false) | result stays gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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
			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("scalar tensor | SumAlong(-1) | returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.SumAlong(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "SumAlong input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor | SumAlong(0) | returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.SumAlong(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "SumAlong input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1], 0) tensor | SumAlong(1) | returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.SumAlong(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "SumAlong input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([3,1], 0) tensor | SumAlong(2) | returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.SumAlong(2)
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

		// ============================== main functionalities ==============================

		t.Run("Full([1], 1) tensor | MaxAlong(0) | returns scalar 1", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MaxAlong(0)
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

		t.Run("Full([3], 1) tensor | MaxAlong(0) | returns scalar 1", func(t *testing.T) {
			x, err := tensor.Full([]int{3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MaxAlong(0)
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

		t.Run("tensor shape [2,3,3] | MaxAlong(0) | returns element-wise max along dim 0", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.MaxAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{8., 2., 0.},
				{5., 4., 3.},
				{7., -3., 7.},
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

		t.Run("tensor shape [2,3,3] | MaxAlong(1) | returns element-wise max along dim 1", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.MaxAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{7., 2., 7.},
				{8., 4., 5.},
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

		t.Run("tensor shape [2,3,3] | MaxAlong(2) | returns element-wise max along dim 2", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.MaxAlong(2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{2., 3., 7.},
				{8., 5., 5.},
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

		t.Run("[3,4] tensor | MaxAlong(0) then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MaxAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | MaxAlong(0) | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | MaxAlong(0) | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MaxAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | MaxAlong(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MaxAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | MaxAlong(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | MaxAlong(0) then BackPropagate | y has nil gradient", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("grad-tracked [2,4,4] tensor with rows 1/2/3/4 | MaxAlong(1) then BackPropagate | gradient concentrates on max row", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [1,4] tensor with three max-ties | MaxAlong(1) then BackPropagate | gradient is divided evenly among tied max positions", func(t *testing.T) {
			x, err := tensor.Of([][]float64{{3., 5., 5., 5.}}, &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Of([][]float64{{0., 1. / 3., 1. / 3., 1. / 3.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [3,2,2] tensor with batch values 1/3/2 | MaxAlong(0) then BackPropagate | gradient is 1 at batch-max positions, 0 elsewhere", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1., 1.},
					{1., 1.},
				},
				{
					{3., 3.},
					{3., 3.},
				},
				{
					{2., 2.},
					{2., 2.},
				},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
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

			g := x.Gradient()

			h, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
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

		t.Run("large [1,2^20] tensor filled with 7 | MaxAlong(1) | returns Full([1], 7)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MaxAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor with all-tied values | MaxAlong(1) then BackPropagate | gradient is divided evenly among all tied positions [1,2^20]", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{1, n}, 1./float64(n), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor | concurrent repeated MaxAlong(1) over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x.MaxAlong(1)
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

		t.Run("grad-tracked [3,4] tensor | MaxAlong(0) then ResetGradient(source, false) | result stays gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MaxAlong(0)
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

		// ============================== validations ==============================

		t.Run("scalar tensor | MaxAlong(-1) | returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.MaxAlong(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "MaxAlong input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor | MaxAlong(0) | returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.MaxAlong(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "MaxAlong input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1], 0) tensor | MaxAlong(1) | returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.MaxAlong(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "MaxAlong input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([3,1], 0) tensor | MaxAlong(2) | returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.MaxAlong(2)
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

		// ============================== main functionalities ==============================

		t.Run("Full([1], 1) tensor | MinAlong(0) | returns scalar 1", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MinAlong(0)
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

		t.Run("Full([3], 1) tensor | MinAlong(0) | returns scalar 1", func(t *testing.T) {
			x, err := tensor.Full([]int{3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MinAlong(0)
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

		t.Run("tensor shape [2,3,3] | MinAlong(0) | returns element-wise min along dim 0", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.MinAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{1., -1., -5.},
				{0., -1., -3.},
				{1., -7., 5.},
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

		t.Run("tensor shape [2,3,3] | MinAlong(1) | returns element-wise min along dim 1", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.MinAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{0., -7., -5.},
				{1., -3., -3.},
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

		t.Run("tensor shape [2,3,3] | MinAlong(2) | returns element-wise min along dim 2", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.MinAlong(2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{-5., -1., -7.},
				{-1., -3., -3.},
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

		t.Run("[3,4] tensor | MinAlong(0) then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MinAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | MinAlong(0) | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | MinAlong(0) | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | MinAlong(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | MinAlong(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | MinAlong(0) then BackPropagate | y has nil gradient", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("grad-tracked [2,4,4] tensor with rows 1/2/3/4 | MinAlong(1) then BackPropagate | gradient concentrates on min row", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [1,4] tensor with three min-ties | MinAlong(1) then BackPropagate | gradient is divided evenly among tied min positions", func(t *testing.T) {
			x, err := tensor.Of([][]float64{{5., 3., 3., 3.}}, &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Of([][]float64{{0., 1. / 3., 1. / 3., 1. / 3.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [3,2,2] tensor with batch values 2/1/3 | MinAlong(0) then BackPropagate | gradient is 1 at batch-min positions, 0 elsewhere", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
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

		t.Run("large [1,2^20] tensor filled with 7 | MinAlong(1) | returns Full([1], 7)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MinAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor with all-tied values | MinAlong(1) then BackPropagate | gradient is divided evenly among all tied positions [1,2^20]", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{1, n}, 1./float64(n), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor | concurrent repeated MinAlong(1) over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x.MinAlong(1)
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

		t.Run("grad-tracked [3,4] tensor | MinAlong(0) then ResetGradient(source, false) | result stays gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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
			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("scalar tensor | MinAlong(-1) | returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.MinAlong(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "MinAlong input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor | MinAlong(0) | returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.MinAlong(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "MinAlong input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1], 0) tensor | MinAlong(1) | returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.MinAlong(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "MinAlong input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([3,1], 0) tensor | MinAlong(2) | returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.MinAlong(2)
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

		// ============================== main functionalities ==============================

		t.Run("Full([1], 1) tensor | AvgAlong(0) | returns scalar 1", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(0)
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

		t.Run("Full([3], 2) tensor | AvgAlong(0) | returns scalar 2", func(t *testing.T) {
			x, err := tensor.Full([]int{3}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(0)
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

		t.Run("Full([3,4,5], 3) tensor | AvgAlong(0) | returns Full([4,5], 3)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4, 5}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{4, 5}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,4,5], 4) tensor | AvgAlong(1) | returns Full([3,5], 4)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4, 5}, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3, 5}, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,4,5], 5) tensor | AvgAlong(2) | returns Full([3,4], 5)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4, 5}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3, 4}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,2,2,2]) tensor | AvgAlong(1) | returns Of([2,2,2])", func(t *testing.T) {
			x, err := tensor.Of([][][][]float64{
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

			y, err := x.AvgAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
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

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3,2,2]) tensor | AvgAlong(3) | returns Of([2,3,2])", func(t *testing.T) {
			x, err := tensor.Of([][][][]float64{
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

			y, err := x.AvgAlong(3)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
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

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1e308, 1e308, 1e308] | AvgAlong(0) | returns 1e308 despite overflow in naive sum", func(t *testing.T) {
			x, err := tensor.Of([]float64{1e308, 1e308, 1e308}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(1e308, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D large tensor with 1000 elements each 1e306 | AvgAlong(0) | returns 1e306 despite overflow in naive sum", func(t *testing.T) {
			data := make([]float64, 1000)
			for i := range data {
				data[i] = 1e306
			}

			x, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(1e306, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D tensor shape [3,2,4] with 1e308 | AvgAlong(0) | returns Full([2,4], 1e308) despite overflow in naive sum", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1e308, 1e308, 1e308, 1e308},
					{1e308, 1e308, 1e308, 1e308},
				},
				{
					{1e308, 1e308, 1e308, 1e308},
					{1e308, 1e308, 1e308, 1e308},
				},
				{
					{1e308, 1e308, 1e308, 1e308},
					{1e308, 1e308, 1e308, 1e308},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{2, 4}, 1e308, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | AvgAlong(0) then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | AvgAlong(0) | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | AvgAlong(0) | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | AvgAlong(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | AvgAlong(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | AvgAlong(0) then BackPropagate | y has nil gradient", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("grad-tracked [2,4,4] tensor | AvgAlong(1) then BackPropagate | gradient is 0.25 everywhere", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 4, 4}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{2, 4, 4}, 0.25, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [2,3,4] tensor | AvgAlong(2) then BackPropagate | gradient is 0.25 everywhere", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 3, 4}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{2, 3, 4}, 0.25, &tensor.Config{Device: dev})
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

		t.Run("large [1,2^20] tensor filled with 7 | AvgAlong(1) | returns Full([1], 7)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor | AvgAlong(1) then BackPropagate | gradient of x is 1/2^20 everywhere [1,2^20]", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{1, n}, 1./float64(n), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor | concurrent repeated AvgAlong(1) over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x.AvgAlong(1)
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

		t.Run("grad-tracked [3,4] tensor | AvgAlong(0) then ResetGradient(source, false) | result stays gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.AvgAlong(0)
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

		// ============================== validations ==============================

		t.Run("scalar tensor | AvgAlong(-1) | returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.AvgAlong(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "AvgAlong input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor | AvgAlong(0) | returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.AvgAlong(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "AvgAlong input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1], 0) tensor | AvgAlong(1) | returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.AvgAlong(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "AvgAlong input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([3,1], 0) tensor | AvgAlong(2) | returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.AvgAlong(2)
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

		// ============================== main functionalities ==============================

		t.Run("Full([1], 1) tensor | VarAlong(0) | returns scalar 0", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(0)
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

		t.Run("Full([3], 5) tensor | VarAlong(0) | returns scalar 0", func(t *testing.T) {
			x, err := tensor.Full([]int{3}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(0)
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

		t.Run("tensor shape [3] | VarAlong(0) | returns scalar 9", func(t *testing.T) {
			x, err := tensor.Of([]float64{-3., 0., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("tensor shape [3,3] | VarAlong(0) | returns [3] tensor with all 9", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
				{7., 8., 9.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{9., 9., 9.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("tensor shape [3,2,2] | VarAlong(0) | returns [2,2] tensor with all 1", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.VarAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{1., 1.},
				{1., 1.},
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

		t.Run("tensor shape [3,2,2] | VarAlong(1) | returns [3,2] tensor with all 18", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.VarAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{18., 18.},
				{18., 18.},
				{18., 18.},
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

		t.Run("tensor shape [3,2,2] | VarAlong(2) | returns [3,2] tensor with all 4.5", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.VarAlong(2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{4.5, 4.5},
				{4.5, 4.5},
				{4.5, 4.5},
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

		t.Run("tensor shape [1,2,4,5] | VarAlong(3) | returns [1,2,4] tensor", func(t *testing.T) {
			x, err := tensor.Of([][][][]float64{
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

			y, err := x.VarAlong(3)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
				{
					{10., 40., 90., 160.},
					{250., 360., 490., 640.},
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

		t.Run("1D tensor [1e8+3, 1e8, 1e8-3] | VarAlong(0) | returns scalar 9 without catastrophic cancellation", func(t *testing.T) {
			x, err := tensor.Of([]float64{1e8 + 3, 1e8, 1e8 - 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(9., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D large tensor with 1000 elements each 1e155 | VarAlong(0) | returns scalar 0 despite x^2 overflow in naive computation", func(t *testing.T) {
			data := make([]float64, 1000)
			for i := range data {
				data[i] = 1e155
			}

			x, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(0)
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

		t.Run("3D tensor shape [2,3,4] with ±3 offset around 1e8 | VarAlong(0) | returns Full([3,4], 18) without catastrophic cancellation", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1e8 + 3, 1e8 + 3, 1e8 + 3, 1e8 + 3},
					{1e8 + 3, 1e8 + 3, 1e8 + 3, 1e8 + 3},
					{1e8 + 3, 1e8 + 3, 1e8 + 3, 1e8 + 3},
				},
				{
					{1e8 - 3, 1e8 - 3, 1e8 - 3, 1e8 - 3},
					{1e8 - 3, 1e8 - 3, 1e8 - 3, 1e8 - 3},
					{1e8 - 3, 1e8 - 3, 1e8 - 3, 1e8 - 3},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3, 4}, 18., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | VarAlong(0) then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | VarAlong(0) | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | VarAlong(0) | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | VarAlong(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | VarAlong(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | VarAlong(0) then BackPropagate | y has nil gradient", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("grad-tracked Full([5,1,3], 1) tensor | VarAlong(1) then BackPropagate | gradient of x is all-zeros [5,1,3]", func(t *testing.T) {
			x, err := tensor.Full([]int{5, 1, 3}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{5, 1, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [2,3,4] tensor with rows 1/2/3 | VarAlong(1) then BackPropagate | gradient reflects variance derivative", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [2,4,3] tensor with rows 1/2/3/4 | VarAlong(1) then BackPropagate | gradient reflects variance derivative analytically", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [2,1,3] tensor | VarAlong(1) on single-element groups then BackPropagate | gradient is all-zeros [2,1,3]", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 1, 3}, 5., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{2, 1, 3}, 0., &tensor.Config{Device: dev})
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

		t.Run("large [1,2^20] tensor filled with 7 | VarAlong(1) | returns Full([1], 0)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor | VarAlong(1) then BackPropagate | gradient of x is all-zeros [1,2^20]", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
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

		t.Run("[1,2^10] tensor | concurrent repeated VarAlong(1) over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x.VarAlong(1)
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

		t.Run("grad-tracked [3,4] tensor | VarAlong(0) then ResetGradient(source, false) | result stays gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.VarAlong(0)
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

		// ============================== validations ==============================

		t.Run("scalar tensor | VarAlong(-1) | returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.VarAlong(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "VarAlong input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor | VarAlong(0) | returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.VarAlong(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "VarAlong input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1], 0) tensor | VarAlong(1) | returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.VarAlong(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "VarAlong input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([3,1], 0) tensor | VarAlong(2) | returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.VarAlong(2)
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

		// ============================== main functionalities ==============================

		t.Run("Full([1], 1) tensor | StdAlong(0) | returns scalar 0", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(0)
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

		t.Run("Full([3], 5) tensor | StdAlong(0) | returns scalar 0", func(t *testing.T) {
			x, err := tensor.Full([]int{3}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(0)
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

		t.Run("tensor shape [3] | StdAlong(0) | returns scalar 3", func(t *testing.T) {
			x, err := tensor.Of([]float64{-3., 0., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("tensor shape [3,3] | StdAlong(0) | returns [3] tensor with all 3", func(t *testing.T) {
			x, err := tensor.Of([][]float64{
				{1., 2., 3.},
				{4., 5., 6.},
				{7., 8., 9.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([]float64{3., 3., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("tensor shape [3,2,2] | StdAlong(0) | returns [2,2] tensor with all 1", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.StdAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{1., 1.},
				{1., 1.},
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

		t.Run("tensor shape [3,2,2] | StdAlong(1) | returns [3,2] tensor with all sqrt(18)", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.StdAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{math.Sqrt(18.), math.Sqrt(18.)},
				{math.Sqrt(18.), math.Sqrt(18.)},
				{math.Sqrt(18.), math.Sqrt(18.)},
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

		t.Run("tensor shape [3,2,2] | StdAlong(2) | returns [3,2] tensor with all sqrt(4.5)", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.StdAlong(2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{math.Sqrt(4.5), math.Sqrt(4.5)},
				{math.Sqrt(4.5), math.Sqrt(4.5)},
				{math.Sqrt(4.5), math.Sqrt(4.5)},
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

		t.Run("tensor shape [1,2,4,5] | StdAlong(3) | returns [1,2,4] tensor", func(t *testing.T) {
			x, err := tensor.Of([][][][]float64{
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

			y, err := x.StdAlong(3)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
				{
					{math.Sqrt(10.), math.Sqrt(40.), math.Sqrt(90.), math.Sqrt(160.)},
					{math.Sqrt(250.), math.Sqrt(360.), math.Sqrt(490.), math.Sqrt(640.)},
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

		t.Run("1D tensor [1e8+3, 1e8, 1e8-3] | StdAlong(0) | returns scalar 3 without catastrophic cancellation", func(t *testing.T) {
			x, err := tensor.Of([]float64{1e8 + 3, 1e8, 1e8 - 3}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D large tensor with 1000 elements each 1e155 | StdAlong(0) | returns scalar 0 despite x^2 overflow in naive computation", func(t *testing.T) {
			data := make([]float64, 1000)
			for i := range data {
				data[i] = 1e155
			}

			x, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(0)
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

		t.Run("3D tensor shape [2,3,4] with ±3 offset around 1e8 | StdAlong(0) | returns Full([3,4], sqrt(18)) without catastrophic cancellation", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1e8 + 3, 1e8 + 3, 1e8 + 3, 1e8 + 3},
					{1e8 + 3, 1e8 + 3, 1e8 + 3, 1e8 + 3},
					{1e8 + 3, 1e8 + 3, 1e8 + 3, 1e8 + 3},
				},
				{
					{1e8 - 3, 1e8 - 3, 1e8 - 3, 1e8 - 3},
					{1e8 - 3, 1e8 - 3, 1e8 - 3, 1e8 - 3},
					{1e8 - 3, 1e8 - 3, 1e8 - 3, 1e8 - 3},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3, 4}, math.Sqrt(18.), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | StdAlong(0) then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | StdAlong(0) | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | StdAlong(0) | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | StdAlong(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | StdAlong(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | StdAlong(0) then BackPropagate | y has nil gradient", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("grad-tracked Full([5,1,3], 1) tensor | StdAlong(1) then BackPropagate | gradient of x is all-zeros [5,1,3]", func(t *testing.T) {
			x, err := tensor.Full([]int{5, 1, 3}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{5, 1, 3}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [2,3,4] tensor with rows 1/2/3 | StdAlong(1) then BackPropagate | gradient is -0.5/0/0.5 pattern", func(t *testing.T) {
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

			g := x.Gradient()

			h, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [2,4,3] tensor with asymmetric group 1/1/1/5 | StdAlong(1) then BackPropagate | gradient is -1/6 on the three equal rows and 1/2 on the outlier", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1., 1., 1.},
					{1., 1., 1.},
					{1., 1., 1.},
					{5., 5., 5.},
				},
				{
					{1., 1., 1.},
					{1., 1., 1.},
					{1., 1., 1.},
					{5., 5., 5.},
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

			g := x.Gradient()

			h, err := tensor.Of([][][]float64{
				{
					{-1. / 6., -1. / 6., -1. / 6.},
					{-1. / 6., -1. / 6., -1. / 6.},
					{-1. / 6., -1. / 6., -1. / 6.},
					{1. / 2., 1. / 2., 1. / 2.},
				},
				{
					{-1. / 6., -1. / 6., -1. / 6.},
					{-1. / 6., -1. / 6., -1. / 6.},
					{-1. / 6., -1. / 6., -1. / 6.},
					{1. / 2., 1. / 2., 1. / 2.},
				},
			}, &tensor.Config{Device: dev})
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

		t.Run("large [1,2^20] tensor filled with 7 | StdAlong(1) | returns Full([1], 0)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor | concurrent repeated StdAlong(1) over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x.StdAlong(1)
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

		t.Run("grad-tracked [3,4] tensor | StdAlong(0) then ResetGradient(source, false) | result stays gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.StdAlong(0)
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

		// ============================== validations ==============================

		t.Run("scalar tensor | StdAlong(-1) | returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.StdAlong(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "StdAlong input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor | StdAlong(0) | returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.StdAlong(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "StdAlong input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1], 0) tensor | StdAlong(1) | returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.StdAlong(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "StdAlong input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([3,1], 0) tensor | StdAlong(2) | returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.StdAlong(2)
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

		// ============================== main functionalities ==============================

		t.Run("Full([1], 1) tensor | MeanAlong(0) | returns scalar 1", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MeanAlong(0)
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

		t.Run("Full([3], 2) tensor | MeanAlong(0) | returns scalar 2", func(t *testing.T) {
			x, err := tensor.Full([]int{3}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MeanAlong(0)
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

		t.Run("Full([3,4,5], 3) tensor | MeanAlong(0) | returns Full([4,5], 3)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4, 5}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MeanAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{4, 5}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,4,5], 4) tensor | MeanAlong(1) | returns Full([3,5], 4)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4, 5}, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MeanAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3, 5}, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([3,4,5], 5) tensor | MeanAlong(2) | returns Full([3,4], 5)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4, 5}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MeanAlong(2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{3, 4}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,2,2,2]) tensor | MeanAlong(1) | returns Of([2,2,2])", func(t *testing.T) {
			x, err := tensor.Of([][][][]float64{
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

			y, err := x.MeanAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
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

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Of([2,3,2,2]) tensor | MeanAlong(3) | returns Of([2,3,2])", func(t *testing.T) {
			x, err := tensor.Of([][][][]float64{
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

			y, err := x.MeanAlong(3)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
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

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D tensor [1e308, 1e308, 1e308] | MeanAlong(0) | returns 1e308 despite overflow in naive sum", func(t *testing.T) {
			x, err := tensor.Of([]float64{1e308, 1e308, 1e308}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MeanAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(1e308, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("1D large tensor with 1000 elements each 1e306 | MeanAlong(0) | returns 1e306 despite overflow in naive sum", func(t *testing.T) {
			data := make([]float64, 1000)
			for i := range data {
				data[i] = 1e306
			}

			x, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MeanAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of(1e306, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("3D tensor shape [3,2,4] with 1e308 | MeanAlong(0) | returns Full([2,4], 1e308) despite overflow in naive sum", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
				{
					{1e308, 1e308, 1e308, 1e308},
					{1e308, 1e308, 1e308, 1e308},
				},
				{
					{1e308, 1e308, 1e308, 1e308},
					{1e308, 1e308, 1e308, 1e308},
				},
				{
					{1e308, 1e308, 1e308, 1e308},
					{1e308, 1e308, 1e308, 1e308},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MeanAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{2, 4}, 1e308, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | MeanAlong(0) then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MeanAlong(0)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | MeanAlong(0) | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | MeanAlong(0) | y is gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | MeanAlong(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | MeanAlong(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | MeanAlong(0) then BackPropagate | y has nil gradient", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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

		t.Run("grad-tracked [2,4,4] tensor | MeanAlong(1) then BackPropagate | gradient is 0.25 everywhere", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 4, 4}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{2, 4, 4}, 0.25, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("grad-tracked [4,3] tensor | MeanAlong(0) then BackPropagate | gradient is 0.25 everywhere", func(t *testing.T) {
			x, err := tensor.Full([]int{4, 3}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{4, 3}, 0.25, &tensor.Config{Device: dev})
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

		t.Run("large [1,2^20] tensor filled with 7 | MeanAlong(1) | returns Full([1], 7)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.MeanAlong(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("large grad-tracked [1,2^20] tensor | MeanAlong(1) then BackPropagate | gradient of x is 1/2^20 everywhere [1,2^20]", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
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

			g := x.Gradient()

			h, err := tensor.Full([]int{1, n}, 1./float64(n), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor | concurrent repeated MeanAlong(1) over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x.MeanAlong(1)
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

		t.Run("grad-tracked [3,4] tensor | MeanAlong(0) then ResetGradient(source, false) | result stays gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
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
			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to still be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("scalar tensor | MeanAlong(-1) | returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.MeanAlong(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "MeanAlong input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor | MeanAlong(0) | returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.MeanAlong(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "MeanAlong input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1], 0) tensor | MeanAlong(1) | returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.MeanAlong(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "MeanAlong input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([3,1], 0) tensor | MeanAlong(2) | returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.MeanAlong(2)
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

		// ============================== main functionalities ==============================

		t.Run("Full([1], 1) tensor | Argmax(0) | returns scalar 0", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmax(0)
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

		t.Run("Full([3], 1) tensor | Argmax(0) | returns scalar 0", func(t *testing.T) {
			x, err := tensor.Full([]int{3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmax(0)
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

		t.Run("tensor shape [2,3,3] | Argmax(0) | returns argmax indices along dim 0", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.Argmax(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{1., 0., 1.},
				{1., 1., 0.},
				{0., 1., 0.},
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

		t.Run("tensor shape [2,3,3] | Argmax(1) | returns argmax indices along dim 1", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.Argmax(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{2., 0., 2.},
				{0., 1., 2.},
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

		t.Run("tensor shape [2,3,3] | Argmax(2) | returns argmax indices along dim 2", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.Argmax(2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{1., 2., 0.},
				{0., 0., 2.},
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

		t.Run("tensor shape [2,4,2,2] | Argmax(1) | returns argmax indices along dim 1", func(t *testing.T) {
			x, err := tensor.Of([][][][]float64{
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

			y, err := x.Argmax(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
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

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Argmax(0) then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmax(0)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | Argmax(0) | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmax(0)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | Argmax(0) | y is not gradient-tracked because operation is not differentiable", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmax(0)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | Argmax(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmax(0)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Argmax(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmax(0)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Argmax(0) then BackPropagate | y has nil gradient", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmax(0)
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

		t.Run("grad-tracked [3,4] tensor | Argmax(0) then BackPropagate | y is still not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmax(0)
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

		t.Run("large [1,2^20] tensor filled with 7 | Argmax(1) | returns Full([1], 0)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmax(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor | concurrent repeated Argmax(1) over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x.Argmax(1)
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

		t.Run("grad-tracked [3,4] tensor | Argmax(0) then ResetGradient(source, false) | result stays not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmax(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("scalar tensor | Argmax(-1) | returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Argmax(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "Argmax input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor | Argmax(0) | returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Argmax(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "Argmax input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1], 0) tensor | Argmax(1) | returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Argmax(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "Argmax input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([3,1], 0) tensor | Argmax(2) | returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Argmax(2)
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

		// ============================== main functionalities ==============================

		t.Run("Full([1], 1) tensor | Argmin(0) | returns scalar 0", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmin(0)
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

		t.Run("Full([3], 1) tensor | Argmin(0) | returns scalar 0", func(t *testing.T) {
			x, err := tensor.Full([]int{3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmin(0)
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

		t.Run("tensor shape [2,3,3] | Argmin(0) | returns argmin indices along dim 0", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.Argmin(0)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{0., 1., 0.},
				{0., 0., 1.},
				{1., 0., 1.},
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

		t.Run("tensor shape [2,3,3] | Argmin(1) | returns argmin indices along dim 1", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.Argmin(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{1., 2., 0.},
				{2., 2., 1.},
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

		t.Run("tensor shape [2,3,3] | Argmin(2) | returns argmin indices along dim 2", func(t *testing.T) {
			x, err := tensor.Of([][][]float64{
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

			y, err := x.Argmin(2)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][]float64{
				{2., 1., 1.},
				{1., 2., 1.},
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

		t.Run("tensor shape [2,4,2,2] | Argmin(1) | returns argmin indices along dim 1", func(t *testing.T) {
			x, err := tensor.Of([][][][]float64{
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

			y, err := x.Argmin(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Of([][][]float64{
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

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[3,4] tensor | Argmin(0) then Device() | returns the device the input was created on", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmin(0)
			if err != nil {
				t.Fatal(err)
			}

			if d := y.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("untracked [3,4] tensor | Argmin(0) | y is not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmin(0)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("grad-tracked [3,4] tensor | Argmin(0) | y is not gradient-tracked because operation is not differentiable", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmin(0)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("grad-tracked [3,4] tensor | Argmin(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmin(0)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Argmin(0) | Gradient() returns nil", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmin(0)
			if err != nil {
				t.Fatal(err)
			}

			if y.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}
		})

		t.Run("untracked [3,4] tensor | Argmin(0) then BackPropagate | y has nil gradient", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmin(0)
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

		t.Run("grad-tracked [3,4] tensor | Argmin(0) then BackPropagate | y is still not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmin(0)
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

		t.Run("large [1,2^20] tensor filled with 7 | Argmin(1) | returns Full([1], 0)", func(t *testing.T) {
			n := 1 << 20

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmin(1)
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := y.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("[1,2^10] tensor | concurrent repeated Argmin(1) over every iteration | never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			x, err := tensor.Full([]int{1, n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			h, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						y, err := x.Argmin(1)
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

		t.Run("grad-tracked [3,4] tensor | Argmin(0) then ResetGradient(source, false) | result stays not gradient-tracked", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := x.Argmin(0)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(x, false)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("scalar tensor | Argmin(-1) | returns error: dimension -1 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Argmin(-1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (-1) being out of range")
			} else if err.Error() != "Argmin input dimension validation failed: expected dimension to be in range [0,0): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("scalar tensor | Argmin(0) | returns error: dimension 0 out of range [0,0)", func(t *testing.T) {
			x, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Argmin(0)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (0) being out of range")
			} else if err.Error() != "Argmin input dimension validation failed: expected dimension to be in range [0,0): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([1], 0) tensor | Argmin(1) | returns error: dimension 1 out of range [0,1)", func(t *testing.T) {
			x, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Argmin(1)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (1) being out of range")
			} else if err.Error() != "Argmin input dimension validation failed: expected dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Full([3,1], 0) tensor | Argmin(2) | returns error: dimension 2 out of range [0,2)", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = x.Argmin(2)
			if err == nil {
				t.Fatal("expected error because of reduced dimension (2) being out of range")
			} else if err.Error() != "Argmin input dimension validation failed: expected dimension to be in range [0,2): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
