package layers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestConcat(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("1D / Forward([-1,2,-3], [4,-5]) mixed tracking, Dim=0 / output equals [-1,2,-3,4,-5]", func(t *testing.T) {
			layer, err := layers.NewConcat(&layers.ConcatConfig{Dim: 0})
			if err != nil {
				t.Fatal(err)
			}

			x1, err := tensor.Of([]float64{-1., 2., -3.}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{4., -5.}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x1, x2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([]float64{-1., 2., -3., 4., -5.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D / Forward([2,2], [2,3]) mixed tracking, Dim=1 / output equals [2,5]", func(t *testing.T) {
			layer, err := layers.NewConcat(&layers.ConcatConfig{Dim: 1})
			if err != nil {
				t.Fatal(err)
			}

			x1, err := tensor.Of([][]float64{
				{-1., 2.},
				{3., -4.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][]float64{
				{5., -6., 7.},
				{-8., 9., -10.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x1, x2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{-1., 2., 5., -6., 7.},
				{3., -4., -8., 9., -10.},
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

		t.Run("3D / Forward three [1,2,1] tensors mixed tracking, Dim=0 / output equals [3,2,1]", func(t *testing.T) {
			layer, err := layers.NewConcat(&layers.ConcatConfig{Dim: 0})
			if err != nil {
				t.Fatal(err)
			}

			x1, err := tensor.Of([][][]float64{{{-5.}, {1.}}}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][][]float64{{{-9.}, {2.}}}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x3, err := tensor.Of([][][]float64{{{-4.}, {7.}}}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x1, x2, x3)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][][]float64{
				{{-5.}, {1.}},
				{{-9.}, {2.}},
				{{-4.}, {7.}},
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

		t.Run("NewConcat(nil) / returns error: nil config", func(t *testing.T) {
			_, err := layers.NewConcat(nil)
			if err == nil {
				t.Fatal("expected error because of nil input config")
			} else if err.Error() != "Concat config data validation failed: expected config not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewConcat with negative Dim / returns error: expected Dim not to be negative", func(t *testing.T) {
			_, err := layers.NewConcat(&layers.ConcatConfig{Dim: -1})
			if err == nil {
				t.Fatal("expected error because of negative 'Dim'")
			} else if err.Error() != "Concat config data validation failed: expected 'Dim' not to be negative: got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("no input tensors / Forward() / returns error: expected at least two input tensors", func(t *testing.T) {
			layer, err := layers.NewConcat(&layers.ConcatConfig{Dim: 0})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward()
			if err == nil {
				t.Fatal("expected error because of not receiving at least two input tensors")
			} else if err.Error() != "Concat input data validation failed: expected at least two input tensors: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("one input tensor / Forward(x) / returns error: expected at least two input tensors", func(t *testing.T) {
			layer, err := layers.NewConcat(&layers.ConcatConfig{Dim: 0})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x)
			if err == nil {
				t.Fatal("expected error because of not receiving at least two input tensors")
			} else if err.Error() != "Concat input data validation failed: expected at least two input tensors: got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
