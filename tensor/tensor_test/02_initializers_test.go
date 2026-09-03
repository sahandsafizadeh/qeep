package tensor_test

import (
	"fmt"
	"sync"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestZeros(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("Zeros(nil) scalar / Equals Full(nil, 0.) / returns true", func(t *testing.T) {
			act, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([1]) 1-element 1D tensor / Equals Full([1], 0.) / returns true", func(t *testing.T) {
			act, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{1}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([3,4]) 2D tensor / Equals Full([3,4], 0.) / returns true", func(t *testing.T) {
			act, err := tensor.Zeros([]int{3, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([2,3,4]) 3D tensor / Equals Full([2,3,4], 0.) / returns true", func(t *testing.T) {
			act, err := tensor.Zeros([]int{2, 3, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{2, 3, 4}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros(nil) scalar tensor / Device() / returns the device it was created on", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if d := ten.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("Zeros(nil) with GradTrack true / GradientTracked() / returns true", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			if !ten.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		t.Run("Zeros(nil) with GradTrack false / GradientTracked() / returns false", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if ten.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		t.Run("Zeros(nil) with nil config / Device() and GradientTracked() / returns CPU and false", func(t *testing.T) {
			ten, err := tensor.Zeros(nil, nil)
			if err != nil {
				t.Fatal(err)
			}

			if d := ten.Device(); d != tensor.CPU {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", tensor.CPU, d)
			}
			if ten.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("Zeros([2^20]) large 1D tensor / Equals Full([2^20], 0.) / returns true", func(t *testing.T) {
			n := 1 << 20

			act, err := tensor.Zeros([]int{n}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{n}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros([2^10]) 1D tensor / concurrent repeated Zeros then Equals over every iteration / never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			exp, err := tensor.Full([]int{n}, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						act, err := tensor.Zeros([]int{n}, &tensor.Config{Device: dev})
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := act.Equals(exp); err != nil {
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

		t.Run("Zeros([3,4]) does not share dims slice / Equals Zeros([3,4]) after mutating dims / returns true", func(t *testing.T) {
			dims := []int{3, 4}

			act, err := tensor.Zeros(dims, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			dims[0] = 1
			dims[1] = 1

			exp, err := tensor.Zeros([]int{3, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Zeros(nil) does not retain config pointer / Device() after mutating config / returns the creation device", func(t *testing.T) {
			conf := &tensor.Config{Device: dev}

			ten, err := tensor.Zeros(nil, conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.Device++

			if d := ten.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("Zeros(nil) does not retain config pointer / GradientTracked() after mutating config / returns the creation setting", func(t *testing.T) {
			conf := &tensor.Config{
				Device:    dev,
				GradTrack: true,
			}

			ten, err := tensor.Zeros(nil, conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.GradTrack = false

			if !ten.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("Zeros([-1]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Zeros([]int{-1}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: Zeros input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([0]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Zeros([]int{0}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: Zeros input dimension validation failed: expected positive dimension sizes: got (0) at position (0)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1,-2]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Zeros([]int{1, -2}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: Zeros input dimension validation failed: expected positive dimension sizes: got (-2) at position (1)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([2,0,1]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Zeros([]int{2, 0, 1}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: Zeros input dimension validation failed: expected positive dimension sizes: got (0) at position (1)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros([1,1,1,1,1,1,1]) / returns error: too many dimensions", func(t *testing.T) {
			_, err := tensor.Zeros([]int{1, 1, 1, 1, 1, 1, 1}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of too many dimensions")
			} else if err.Error() != fmt.Sprintf("%s initialization: Zeros input dimension validation failed: expected at most (6) dimensions: got (7)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Zeros(nil) with invalid device / returns error: invalid device", func(t *testing.T) {
			_, err := tensor.Zeros(nil, &tensor.Config{Device: -1})
			if err == nil {
				t.Fatal("expected error because of invalid input device")
			} else if err.Error() != "Zeros tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestOnes(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("Ones(nil) scalar / Equals Full(nil, 1.) / returns true", func(t *testing.T) {
			act, err := tensor.Ones(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Ones([1]) 1-element 1D tensor / Equals Full([1], 1.) / returns true", func(t *testing.T) {
			act, err := tensor.Ones([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Ones([3,4]) 2D tensor / Equals Full([3,4], 1.) / returns true", func(t *testing.T) {
			act, err := tensor.Ones([]int{3, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Ones([2,3,4]) 3D tensor / Equals Full([2,3,4], 1.) / returns true", func(t *testing.T) {
			act, err := tensor.Ones([]int{2, 3, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{2, 3, 4}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Ones(nil) scalar tensor / Device() / returns the device it was created on", func(t *testing.T) {
			ten, err := tensor.Ones(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if d := ten.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("Ones(nil) with GradTrack true / GradientTracked() / returns true", func(t *testing.T) {
			ten, err := tensor.Ones(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			if !ten.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		t.Run("Ones(nil) with GradTrack false / GradientTracked() / returns false", func(t *testing.T) {
			ten, err := tensor.Ones(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if ten.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		t.Run("Ones(nil) with nil config / Device() and GradientTracked() / returns CPU and false", func(t *testing.T) {
			ten, err := tensor.Ones(nil, nil)
			if err != nil {
				t.Fatal(err)
			}

			if d := ten.Device(); d != tensor.CPU {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", tensor.CPU, d)
			}
			if ten.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("Ones([2^20]) large 1D tensor / Equals Full([2^20], 1.) / returns true", func(t *testing.T) {
			n := 1 << 20

			act, err := tensor.Ones([]int{n}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Ones([2^10]) 1D tensor / concurrent repeated Ones then Equals over every iteration / never errors and always equal", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			exp, err := tensor.Full([]int{n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						act, err := tensor.Ones([]int{n}, &tensor.Config{Device: dev})
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := act.Equals(exp); err != nil {
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

		t.Run("Ones([3,4]) does not share dims slice / Equals Ones([3,4]) after mutating dims / returns true", func(t *testing.T) {
			dims := []int{3, 4}

			act, err := tensor.Ones(dims, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			dims[0] = 1
			dims[1] = 1

			exp, err := tensor.Ones([]int{3, 4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Ones(nil) does not retain config pointer / Device() after mutating config / returns the creation device", func(t *testing.T) {
			conf := &tensor.Config{Device: dev}

			ten, err := tensor.Ones(nil, conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.Device++

			if d := ten.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("Ones(nil) does not retain config pointer / GradientTracked() after mutating config / returns the creation setting", func(t *testing.T) {
			conf := &tensor.Config{
				Device:    dev,
				GradTrack: true,
			}

			ten, err := tensor.Ones(nil, conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.GradTrack = false

			if !ten.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("Ones([-1]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Ones([]int{-1}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: Ones input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Ones([0]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Ones([]int{0}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: Ones input dimension validation failed: expected positive dimension sizes: got (0) at position (0)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Ones([1,-2]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Ones([]int{1, -2}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: Ones input dimension validation failed: expected positive dimension sizes: got (-2) at position (1)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Ones([2,0,1]) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Ones([]int{2, 0, 1}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: Ones input dimension validation failed: expected positive dimension sizes: got (0) at position (1)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Ones([1,1,1,1,1,1,1]) / returns error: too many dimensions", func(t *testing.T) {
			_, err := tensor.Ones([]int{1, 1, 1, 1, 1, 1, 1}, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of too many dimensions")
			} else if err.Error() != fmt.Sprintf("%s initialization: Ones input dimension validation failed: expected at most (6) dimensions: got (7)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Ones(nil) with invalid device / returns error: invalid device", func(t *testing.T) {
			_, err := tensor.Ones(nil, &tensor.Config{Device: -1})
			if err == nil {
				t.Fatal("expected error because of invalid input device")
			} else if err.Error() != "Ones tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestEye(t *testing.T) {

	// ============================== main functionalities ==============================

	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {
		t.Run("Eye(1) 1x1 identity matrix / Equals / returns true", func(t *testing.T) {
			act, err := tensor.Eye(1, &tensor.Config{Device: dev})
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

		t.Run("Eye(2) 2x2 identity matrix / Equals / returns true", func(t *testing.T) {
			act, err := tensor.Eye(2, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{1., 0.},
				{0., 1.},
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

		t.Run("Eye(5) 5x5 identity matrix / Equals / returns true", func(t *testing.T) {
			act, err := tensor.Eye(5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{1., 0., 0., 0., 0.},
				{0., 1., 0., 0., 0.},
				{0., 0., 1., 0., 0.},
				{0., 0., 0., 1., 0.},
				{0., 0., 0., 0., 1.},
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

		t.Run("Eye(1) identity matrix / Device() / returns the device it was created on", func(t *testing.T) {
			ten, err := tensor.Eye(1, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if d := ten.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("Eye(1) with GradTrack true / GradientTracked() / returns true", func(t *testing.T) {
			ten, err := tensor.Eye(1, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			if !ten.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		t.Run("Eye(1) with GradTrack false / GradientTracked() / returns false", func(t *testing.T) {
			ten, err := tensor.Eye(1, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if ten.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		t.Run("Eye(1) with nil config / Device() and GradientTracked() / returns CPU and false", func(t *testing.T) {
			ten, err := tensor.Eye(1, nil)
			if err != nil {
				t.Fatal(err)
			}

			if d := ten.Device(); d != tensor.CPU {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", tensor.CPU, d)
			}
			if ten.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("Eye(2^10) large 2^10x2^10 identity matrix / Equals / returns true", func(t *testing.T) {
			d := 1 << 10

			act, err := tensor.Eye(d, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			data := make([][]float64, d)
			for i := range data {
				data[i] = make([]float64, d)
				data[i][i] = 1.
			}

			exp, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Eye(2^5) 2^5x2^5 identity matrix / concurrent repeated Eye then Equals over every iteration / never errors and always equal", func(t *testing.T) {
			const (
				d  = 1 << 5
				ni = 1 << 4
				ng = 1 << 8
			)

			data := make([][]float64, d)
			for i := range data {
				data[i] = make([]float64, d)
				data[i][i] = 1.
			}

			exp, err := tensor.Of(data, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						act, err := tensor.Eye(d, &tensor.Config{Device: dev})
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := act.Equals(exp); err != nil {
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

		t.Run("Eye(1) does not retain config pointer / Device() after mutating config / returns the creation device", func(t *testing.T) {
			conf := &tensor.Config{Device: dev}

			ten, err := tensor.Eye(1, conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.Device++

			if d := ten.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("Eye(1) does not retain config pointer / GradientTracked() after mutating config / returns the creation setting", func(t *testing.T) {
			conf := &tensor.Config{
				Device:    dev,
				GradTrack: true,
			}

			ten, err := tensor.Eye(1, conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.GradTrack = false

			if !ten.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("Eye(-1) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Eye(-1, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: Eye input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Eye(0) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.Eye(0, &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: Eye input dimension validation failed: expected positive dimension sizes: got (0) at position (0)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Eye(1) with invalid device / returns error: invalid device", func(t *testing.T) {
			_, err := tensor.Eye(1, &tensor.Config{Device: -1})
			if err == nil {
				t.Fatal("expected error because of invalid input device")
			} else if err.Error() != "Eye tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestRandU(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("RandU(nil) scalar tensor / Device() / returns the device it was created on", func(t *testing.T) {
			ten, err := tensor.RandU(nil, 0., 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if d := ten.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("RandU(nil) with GradTrack true / GradientTracked() / returns true", func(t *testing.T) {
			ten, err := tensor.RandU(nil, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			if !ten.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		t.Run("RandU(nil) with GradTrack false / GradientTracked() / returns false", func(t *testing.T) {
			ten, err := tensor.RandU(nil, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if ten.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		t.Run("RandU(nil) with nil config / Device() and GradientTracked() / returns CPU and false", func(t *testing.T) {
			ten, err := tensor.RandU(nil, 0., 1., nil)
			if err != nil {
				t.Fatal(err)
			}

			if d := ten.Device(); d != tensor.CPU {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", tensor.CPU, d)
			}
			if ten.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("RandU([2^20], 0, 1) large 1D tensor / Shape() / returns [2^20]", func(t *testing.T) {
			n := 1 << 20

			_, err := tensor.RandU([]int{n}, 0., 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
		})

		t.Run("RandU([2^10], 0, 1) 1D tensor / concurrent repeated RandU over every iteration / never errors", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						_, err := tensor.RandU([]int{n}, 0., 1., &tensor.Config{Device: dev})
						if err != nil {
							t.Error(err)
							return
						}
					}
				})
			}
			wg.Wait()
		})

		// ============================== side effects ==============================

		t.Run("RandU([3,4], -1, 1) does not share dims slice / Shape() after mutating dims / returns [3,4]", func(t *testing.T) {
			dims := []int{3, 4}

			ten, err := tensor.RandU(dims, -1., 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			dims[0] = 1
			dims[1] = 1

			if _, err := ten.At(2, 3); err != nil {
				t.Fatal(err)
			}
		})

		t.Run("RandU(nil) does not retain config pointer / Device() after mutating config / returns the creation device", func(t *testing.T) {
			conf := &tensor.Config{Device: dev}

			ten, err := tensor.RandU(nil, 0., 1., conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.Device++

			if d := ten.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("RandU(nil) does not retain config pointer / GradientTracked() after mutating config / returns the creation setting", func(t *testing.T) {
			conf := &tensor.Config{
				Device:    dev,
				GradTrack: true,
			}

			ten, err := tensor.RandU(nil, 0., 1., conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.GradTrack = false

			if !ten.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("RandU(nil, 0, -1) / lower bound >= upper bound / returns error: lower bound not less than upper bound", func(t *testing.T) {
			_, err := tensor.RandU(nil, 0., -1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of lower bound not being less than upper bound")
			} else if err.Error() != fmt.Sprintf("%s initialization: RandU random parameter validation failed: expected uniform random lower bound to be less than the upper bound: (0.000000) >= (-1.000000)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandU(nil, 1, 1) / equal bounds / returns error: lower bound not less than upper bound", func(t *testing.T) {
			_, err := tensor.RandU(nil, 1., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of lower bound not being less than upper bound")
			} else if err.Error() != fmt.Sprintf("%s initialization: RandU random parameter validation failed: expected uniform random lower bound to be less than the upper bound: (1.000000) >= (1.000000)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandU([-1], -1, 1) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.RandU([]int{-1}, -1., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: RandU input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandU([0], -1, 1) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.RandU([]int{0}, -1., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: RandU input dimension validation failed: expected positive dimension sizes: got (0) at position (0)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandU([1,-2], -1, 1) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.RandU([]int{1, -2}, -1., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: RandU input dimension validation failed: expected positive dimension sizes: got (-2) at position (1)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandU([2,0,1], -1, 1) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.RandU([]int{2, 0, 1}, -1., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: RandU input dimension validation failed: expected positive dimension sizes: got (0) at position (1)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandU([1,1,1,1,1,1,1], -1, 1) / returns error: too many dimensions", func(t *testing.T) {
			_, err := tensor.RandU([]int{1, 1, 1, 1, 1, 1, 1}, -1., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of too many dimensions")
			} else if err.Error() != fmt.Sprintf("%s initialization: RandU input dimension validation failed: expected at most (6) dimensions: got (7)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandU(nil) with invalid device / returns error: invalid device", func(t *testing.T) {
			_, err := tensor.RandU(nil, 0., 1., &tensor.Config{Device: -1})
			if err == nil {
				t.Fatal("expected error because of invalid input device")
			} else if err.Error() != "RandU tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestRandN(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("RandN(nil) scalar tensor / Device() / returns the device it was created on", func(t *testing.T) {
			ten, err := tensor.RandN(nil, 0., 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if d := ten.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("RandN(nil) with GradTrack true / GradientTracked() / returns true", func(t *testing.T) {
			ten, err := tensor.RandN(nil, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			if !ten.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		t.Run("RandN(nil) with GradTrack false / GradientTracked() / returns false", func(t *testing.T) {
			ten, err := tensor.RandN(nil, 0., 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if ten.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		t.Run("RandN(nil) with nil config / Device() and GradientTracked() / returns CPU and false", func(t *testing.T) {
			ten, err := tensor.RandN(nil, 0., 1., nil)
			if err != nil {
				t.Fatal(err)
			}

			if d := ten.Device(); d != tensor.CPU {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", tensor.CPU, d)
			}
			if ten.GradientTracked() {
				t.Fatal("expected tensor to not be gradient tracked")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("RandN([2^20], 0, 1) large 1D tensor / Shape() / returns [2^20]", func(t *testing.T) {
			n := 1 << 20

			_, err := tensor.RandN([]int{n}, 0., 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
		})

		t.Run("RandN([2^10], 0, 1) 1D tensor / concurrent repeated RandN over every iteration / never errors", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						_, err := tensor.RandN([]int{n}, 0., 1., &tensor.Config{Device: dev})
						if err != nil {
							t.Error(err)
							return
						}
					}
				})
			}
			wg.Wait()
		})

		// ============================== side effects ==============================

		t.Run("RandN([3,4], 0, 1) does not share dims slice / Shape() after mutating dims / returns [3,4]", func(t *testing.T) {
			dims := []int{3, 4}

			ten, err := tensor.RandN(dims, 0., 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			dims[0] = 1
			dims[1] = 1

			if _, err := ten.At(2, 3); err != nil {
				t.Fatal(err)
			}
		})

		t.Run("RandN(nil) does not retain config pointer / Device() after mutating config / returns the creation device", func(t *testing.T) {
			conf := &tensor.Config{Device: dev}

			ten, err := tensor.RandN(nil, 0., 1., conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.Device++

			if d := ten.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("RandN(nil) does not retain config pointer / GradientTracked() after mutating config / returns the creation setting", func(t *testing.T) {
			conf := &tensor.Config{
				Device:    dev,
				GradTrack: true,
			}

			ten, err := tensor.RandN(nil, 0., 1., conf)
			if err != nil {
				t.Fatal(err)
			}

			conf.GradTrack = false

			if !ten.GradientTracked() {
				t.Fatal("expected tensor to be gradient tracked")
			}
		})

		// ============================== validations ==============================

		t.Run("RandN(nil, 0, -1) / negative standard deviation / returns error: std dev not positive", func(t *testing.T) {
			_, err := tensor.RandN(nil, 0., -1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive standard deviation")
			} else if err.Error() != fmt.Sprintf("%s initialization: RandN random parameter validation failed: expected normal random standard deviation to be positive: got (-1.000000)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandN(nil, -1, 0) / zero standard deviation / returns error: std dev not positive", func(t *testing.T) {
			_, err := tensor.RandN(nil, -1., 0., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive standard deviation")
			} else if err.Error() != fmt.Sprintf("%s initialization: RandN random parameter validation failed: expected normal random standard deviation to be positive: got (0.000000)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandN([-1], 0, 1) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.RandN([]int{-1}, 0., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: RandN input dimension validation failed: expected positive dimension sizes: got (-1) at position (0)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandN([0], 0, 1) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.RandN([]int{0}, 0., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: RandN input dimension validation failed: expected positive dimension sizes: got (0) at position (0)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandN([1,-2], 0, 1) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.RandN([]int{1, -2}, 0., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: RandN input dimension validation failed: expected positive dimension sizes: got (-2) at position (1)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandN([2,0,1], 0, 1) / returns error: non-positive dimension", func(t *testing.T) {
			_, err := tensor.RandN([]int{2, 0, 1}, 0., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of non-positive dimension")
			} else if err.Error() != fmt.Sprintf("%s initialization: RandN input dimension validation failed: expected positive dimension sizes: got (0) at position (1)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandN([1,1,1,1,1,1,1], 0, 1) / returns error: too many dimensions", func(t *testing.T) {
			_, err := tensor.RandN([]int{1, 1, 1, 1, 1, 1, 1}, 0., 1., &tensor.Config{Device: dev})
			if err == nil {
				t.Fatal("expected error because of too many dimensions")
			} else if err.Error() != fmt.Sprintf("%s initialization: RandN input dimension validation failed: expected at most (6) dimensions: got (7)", dev) {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("RandN(nil) with invalid device / returns error: invalid device", func(t *testing.T) {
			_, err := tensor.RandN(nil, 0., 1., &tensor.Config{Device: -1})
			if err == nil {
				t.Fatal("expected error because of invalid input device")
			} else if err.Error() != "RandN tensor config data validation failed: invalid input device" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func TestConcat(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("Full([3]) and Full([5]) / Concat(dim=0) / returns Full([8])", func(t *testing.T) {
			t1, err := tensor.Full([]int{3}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{5}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := tensor.Concat([]tensor.Tensor{t1, t2}, 0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{8}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([1,5,3]), Full([3,5,3]), Full([2,5,3]), Full([4,5,3]) / Concat(dim=0) / returns Full([10,5,3])", func(t *testing.T) {
			t1, err := tensor.Full([]int{1, 5, 3}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{3, 5, 3}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Full([]int{2, 5, 3}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t4, err := tensor.Full([]int{4, 5, 3}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := tensor.Concat([]tensor.Tensor{t1, t2, t3, t4}, 0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{10, 5, 3}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Full([4,2,3]), Full([4,4,3]), Full([4,1,3]), Full([4,3,3]) / Concat(dim=1) / returns Full([4,10,3])", func(t *testing.T) {
			t1, err := tensor.Full([]int{4, 2, 3}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{4, 4, 3}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Full([]int{4, 1, 3}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t4, err := tensor.Full([]int{4, 3, 3}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := tensor.Concat([]tensor.Tensor{t1, t2, t3, t4}, 1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{4, 10, 3}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Three copies of 1x2x3 tensor / Concat(dim=0) / returns 3x2x3 tensor", func(t *testing.T) {
			t1, err := tensor.Of([][][]float64{
				{
					{0., 1., 2.},
					{3., 4., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := tensor.Concat([]tensor.Tensor{t1, t1, t1}, 0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{0., 1., 2.},
					{3., 4., 5.},
				},
				{
					{0., 1., 2.},
					{3., 4., 5.},
				},
				{
					{0., 1., 2.},
					{3., 4., 5.},
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

		t.Run("Three copies of 1x2x3 tensor / Concat(dim=1) / returns 1x6x3 tensor", func(t *testing.T) {
			t1, err := tensor.Of([][][]float64{
				{
					{0., 1., 2.},
					{3., 4., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := tensor.Concat([]tensor.Tensor{t1, t1, t1}, 1)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{0., 1., 2.},
					{3., 4., 5.},
					{0., 1., 2.},
					{3., 4., 5.},
					{0., 1., 2.},
					{3., 4., 5.},
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

		t.Run("Three copies of 1x2x3 tensor / Concat(dim=2) / returns 1x2x9 tensor", func(t *testing.T) {
			t1, err := tensor.Of([][][]float64{
				{
					{0., 1., 2.},
					{3., 4., 5.},
				},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := tensor.Concat([]tensor.Tensor{t1, t1, t1}, 2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{
					{0., 1., 2., 0., 1., 2., 0., 1., 2.},
					{3., 4., 5., 3., 4., 5., 3., 4., 5.},
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

		t.Run("Concat([Full([2]), Full([3])], dim=0) / Device() / returns the device the inputs were created on", func(t *testing.T) {
			t1, err := tensor.Full([]int{2}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{3}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := tensor.Concat([]tensor.Tensor{t1, t2}, 0)
			if err != nil {
				t.Fatal(err)
			}

			if d := act.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})

		t.Run("t1 untracked, t2 untracked / Concat along axis 0 / y is not gradient-tracked", func(t *testing.T) {
			t1, err := tensor.Full([]int{1}, 7., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{1}, 7., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Concat([]tensor.Tensor{t1, t2}, 0)
			if err != nil {
				t.Fatal(err)
			}

			if y.GradientTracked() {
				t.Fatal("expected gradient not to be tracked")
			}
		})

		t.Run("t1 grad-tracked, t2 untracked / Concat along axis 0 / y is gradient-tracked", func(t *testing.T) {
			t1, err := tensor.Full([]int{1}, 7., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{1}, 7., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Concat([]tensor.Tensor{t1, t2}, 0)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		t.Run("t1 untracked, t2 grad-tracked / Concat along axis 0 / y is gradient-tracked", func(t *testing.T) {
			t1, err := tensor.Full([]int{1}, 7., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{1}, 7., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Concat([]tensor.Tensor{t1, t2}, 0)
			if err != nil {
				t.Fatal(err)
			}

			if !y.GradientTracked() {
				t.Fatal("expected gradient to be tracked")
			}
		})

		// =============== gradients ===============

		t.Run("two grad-tracked [2,3] tensors / Concat along axis 0 then BackPropagate / each input gradient is all-ones [2,3]", func(t *testing.T) {
			x1, err := tensor.Full([]int{2, 3}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{2, 3}, 3., &tensor.Config{
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

			exp1, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act1.Equals(exp1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := act2.Equals(exp2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("two grad-tracked [3,2] tensors / Concat along last axis (axis 1) then BackPropagate / each input gradient is all-ones [3,2]", func(t *testing.T) {
			x1, err := tensor.Full([]int{3, 2}, 3., &tensor.Config{
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

			exp1, err := tensor.Full([]int{3, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err := tensor.Full([]int{3, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act1.Equals(exp1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := act2.Equals(exp2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("three grad-tracked inputs of shapes [5,1,3] [5,5,3] [5,2,3] / Concat along axis 1 then BackPropagate / each input gradient is all-ones matching its shape", func(t *testing.T) {
			x1, err := tensor.Full([]int{5, 1, 3}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Full([]int{5, 5, 3}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x3, err := tensor.Full([]int{5, 2, 3}, 3., &tensor.Config{
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

			exp1, err := tensor.Full([]int{5, 1, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err := tensor.Full([]int{5, 5, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			exp3, err := tensor.Full([]int{5, 2, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act1.Equals(exp1); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := act2.Equals(exp2); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := act3.Equals(exp3); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("same grad-tracked [3,4] tensor used twice / Concat([x,x], axis 0) then BackPropagate / gradient of x is all-twos [3,4]", func(t *testing.T) {
			x, err := tensor.Full([]int{3, 4}, 3., &tensor.Config{
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

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("four Full([2^20], 7) large tensors / Concat(dim=0) / returns Full([4*2^20], 7)", func(t *testing.T) {
			n := 1 << 20

			t1, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t4, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := tensor.Concat([]tensor.Tensor{t1, t2, t3, t4}, 0)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{4 * n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("four Full([2^10], 7) tensors / concurrent repeated Concat(dim=0) over every iteration / returns Full([4*2^10], 7)", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 1 << 4
				ng = 1 << 8
			)

			t1, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t4, err := tensor.Full([]int{n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{4 * n}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			var wg sync.WaitGroup
			for range ng {
				wg.Go(func() {
					for range ni {
						act, err := tensor.Concat([]tensor.Tensor{t1, t2, t3, t4}, 0)
						if err != nil {
							t.Error(err)
							return
						}

						if eq, err := act.Equals(exp); err != nil {
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

		t.Run("Concat([Full([4], 3), Full([6], 3)], 0) does not share input slice / Concat then mutating slice / returns Full([10], 3)", func(t *testing.T) {
			t1, err := tensor.Full([]int{4}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{6}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			ts := []tensor.Tensor{t1, t2}

			act, err := tensor.Concat(ts, 0)
			if err != nil {
				t.Fatal(err)
			}

			ts[1], err = tensor.Full([]int{6}, 7., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{10}, 3., &tensor.Config{Device: dev})
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

		t.Run("Concat(nil, 0) / returns error: fewer than 2 tensors", func(t *testing.T) {
			_, err := tensor.Concat(nil, 0)
			if err == nil {
				t.Fatal("expected error because of the number of input tensors being less than (2)")
			} else if err.Error() != "Concat tensor implementation validation failed: expected at least (2) tensors: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([nil], 0) / returns error: fewer than 2 tensors", func(t *testing.T) {
			_, err := tensor.Concat([]tensor.Tensor{nil}, 0)
			if err == nil {
				t.Fatal("expected error because of the number of input tensors being less than (2)")
			} else if err.Error() != "Concat tensor implementation validation failed: expected at least (2) tensors: got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Full([2]), nil], 0) / returns error: unsupported tensor implementation", func(t *testing.T) {
			t1, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, nil}, 0)
			if err == nil {
				t.Fatal("expected error because of nil input tensors")
			} else if err.Error() != "Concat tensor implementation validation failed: unsupported tensor implementation" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([scalar, scalar], 0) / returns error: scalar tensor cannot be concatenated", func(t *testing.T) {
			t1, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2}, 0)
			if err == nil {
				t.Fatal("expected error because of having scalar tensors as input")
			} else if err.Error() != "Concat: Concat inputs' dimension validation failed: scalar tensor can not be concatenated: got tensor (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Full([2]), Full([2]), Full([2,2])], 0) / returns error: tensors have different number of dimensions", func(t *testing.T) {
			t1, err := tensor.Full([]int{2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 0)
			if err == nil {
				t.Fatal("expected error because of the input tensors not having equal number of dimensions")
			} else if err.Error() != "Concat: Concat inputs' dimension validation failed: expected tensors to have the same number of dimensions: (2) != (1) for tensor (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Full([1]), Full([3])], -1) / returns error: dimension out of range [0,1)", func(t *testing.T) {
			t1, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2}, -1)
			if err == nil {
				t.Fatal("expected error because of negative dimension")
			} else if err.Error() != "Concat: Concat inputs' dimension validation failed: expected concat dimension to be in range [0,1): got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Full([1]), Full([3])], 1) / returns error: dimension out of range [0,1)", func(t *testing.T) {
			t1, err := tensor.Full([]int{1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2}, 1)
			if err == nil {
				t.Fatal("expected error because of dimension (1) being out of range")
			} else if err.Error() != "Concat: Concat inputs' dimension validation failed: expected concat dimension to be in range [0,1): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Full([3,3]), Full([3,3])], 2) / returns error: dimension out of range [0,2)", func(t *testing.T) {
			t1, err := tensor.Full([]int{3, 3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{3, 3}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2}, 2)
			if err == nil {
				t.Fatal("expected error because of dimension (2) being out of range")
			} else if err.Error() != "Concat: Concat inputs' dimension validation failed: expected concat dimension to be in range [0,2): got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Full([2,2,2]), Full([2,2,1]), Full([3,2,2])], 0) / returns error: size mismatch at dim 2", func(t *testing.T) {
			t1, err := tensor.Full([]int{2, 2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{2, 2, 1}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Full([]int{3, 2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 0)
			if err == nil {
				t.Fatal("expected error because of size mismatch along dimension (2)")
			} else if err.Error() != "Concat: Concat inputs' dimension validation failed: expected tensor sizes to match in all dimensions except (0): (1) != (2) for dimension (2) for tensor (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Full([2,1,2]), Full([2,2,2]), Full([3,2,2])], 0) / returns error: size mismatch at dim 1", func(t *testing.T) {
			t1, err := tensor.Full([]int{2, 1, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{2, 2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Full([]int{3, 2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 0)
			if err == nil {
				t.Fatal("expected error because of size mismatch along dimension (1)")
			} else if err.Error() != "Concat: Concat inputs' dimension validation failed: expected tensor sizes to match in all dimensions except (0): (2) != (1) for dimension (1) for tensor (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Full([2,1,2]), Full([1,2,2]), Full([2,3,2])], 1) / returns error: size mismatch at dim 0", func(t *testing.T) {
			t1, err := tensor.Full([]int{2, 1, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{1, 2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Full([]int{2, 3, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 1)
			if err == nil {
				t.Fatal("expected error because of size mismatch along dimension (0)")
			} else if err.Error() != "Concat: Concat inputs' dimension validation failed: expected tensor sizes to match in all dimensions except (1): (1) != (2) for dimension (0) for tensor (1)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})

	tensor.RunTestLogicCrossDevice(func(d1 tensor.Device, d2 tensor.Device) {

		// ============================== validations ==============================

		t.Run("Concat([Full(d1), Full(d2)], 0) / returns error: source tensors not on the same device", func(t *testing.T) {
			t1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2}, 0)
			if err == nil {
				t.Fatal("expected error because of source tensors not being on the same device")
			} else if err.Error() != "Concat tensor implementation validation failed: input tensors not on the same device" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Concat([Full(d1), Full(d1), Full(d2)], 0) / returns error: source tensors not on the same device", func(t *testing.T) {
			t1, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			t2, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d1})
			if err != nil {
				t.Fatal(err)
			}
			t3, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: d2})
			if err != nil {
				t.Fatal(err)
			}

			_, err = tensor.Concat([]tensor.Tensor{t1, t2, t3}, 0)
			if err == nil {
				t.Fatal("expected error because of source tensors not being on the same device")
			} else if err.Error() != "Concat tensor implementation validation failed: input tensors not on the same device" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
