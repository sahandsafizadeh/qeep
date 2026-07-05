package layers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestAdd(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("scalar (0-dim) / Forward(-4, 2) both untracked / output equals -2", func(t *testing.T) {
			layer := layers.NewAdd()

			x1, err := tensor.Of(-4., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of(2., &tensor.Config{
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

			exp, err := tensor.Of(-2., &tensor.Config{
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

		t.Run("1D / Forward([-1,2,-3], [4,-5,6]) mixed tracking / output equals [3,-3,3]", func(t *testing.T) {
			layer := layers.NewAdd()

			x1, err := tensor.Of([]float64{-1., 2., -3.}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([]float64{4., -5., 6.}, &tensor.Config{
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

			exp, err := tensor.Of([]float64{3., -3., 3.}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("2D / Forward([[-1,2],[3,-4]], [[5,-6],[-7,8]]) mixed tracking / output equals [[4,-4],[-4,4]]", func(t *testing.T) {
			layer := layers.NewAdd()

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
				{5., -6.},
				{-7., 8.},
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
				{4., -4.},
				{-4., 4.},
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

		t.Run("3D / Forward both tracked / output equals [[[-9],[8]],[[-3],[4]]]", func(t *testing.T) {
			layer := layers.NewAdd()

			x1, err := tensor.Of([][][]float64{
				{{-5.}, {1.}},
				{{-9.}, {2.}},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Of([][][]float64{
				{{-4.}, {7.}},
				{{6.}, {2.}},
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

			exp, err := tensor.Of([][][]float64{
				{{-9.}, {8.}},
				{{-3.}, {4.}},
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

		t.Run("broadcasting / Forward(Zeros([3,1,5,1]), Ones([1,2,3,4,1,6])) / output equals Ones([1,2,3,4,5,6])", func(t *testing.T) {
			layer := layers.NewAdd()

			x1, err := tensor.Zeros([]int{3, 1, 5, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			x2, err := tensor.Ones([]int{1, 2, 3, 4, 1, 6}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x1, x2)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Ones([]int{1, 2, 3, 4, 5, 6}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============ validations ============

		t.Run("one input tensor / Forward(x) / returns error: expected exactly two input tensors", func(t *testing.T) {
			layer := layers.NewAdd()

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x)
			if err == nil {
				t.Fatal("expected error because of not receiving two input tensors")
			} else if err.Error() != "Add input data validation failed: expected exactly two input tensors: got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("three input tensors / Forward(x, x, x) / returns error: expected exactly two input tensors", func(t *testing.T) {
			layer := layers.NewAdd()

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x, x, x)
			if err == nil {
				t.Fatal("expected error because of not receiving two input tensors")
			} else if err.Error() != "Add input data validation failed: expected exactly two input tensors: got (3)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
