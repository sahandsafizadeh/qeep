package layers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestDropout(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Dropout(nil) / Forward([[3,1,2]]) without grad track / output equals [[3,1,2]]", func(t *testing.T) {
			layer, err := layers.NewDropout(nil)
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{{3., 1., 2.}}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{{3., 1., 2.}}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Dropout(nil) / Forward(-5) scalar with grad track / output equals 0 or -10", func(t *testing.T) {
			layer, err := layers.NewDropout(nil)
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of(-5., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp1, err := tensor.Of(0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			exp2, err := tensor.Of(-10., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			eq1, err := act.Equals(exp1)
			if err != nil {
				t.Fatal(err)
			}
			eq2, err := act.Equals(exp2)
			if err != nil {
				t.Fatal(err)
			}

			if !eq1 && !eq2 {
				t.Fatalf("expected tensors to be equal")
			}
		})

		t.Run("Dropout(rate=0.9) / Forward([[1]]) with grad track / output near 0 or near 10", func(t *testing.T) {
			layer, err := layers.NewDropout(&layers.DropoutConfig{
				Rate: 0.9,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{{1.}}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl1, err := tensor.Of([][]float64{{-0.1}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu1, err := tensor.Of([][]float64{{0.1}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expl2, err := tensor.Of([][]float64{{9.9}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu2, err := tensor.Of([][]float64{{10.1}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			inRange1 := func() (bool, error) {
				if p, err := act.Gt(expl1); err != nil {
					return false, err
				} else if p.Sum() < float64(p.NElems()) {
					return false, nil
				}
				if p, err := act.Lt(expu1); err != nil {
					return false, err
				} else if p.Sum() < float64(p.NElems()) {
					return false, nil
				}

				return true, nil
			}

			inRange2 := func() (bool, error) {
				if p, err := act.Gt(expl2); err != nil {
					return false, err
				} else if p.Sum() < float64(p.NElems()) {
					return false, nil
				}
				if p, err := act.Lt(expu2); err != nil {
					return false, err
				} else if p.Sum() < float64(p.NElems()) {
					return false, nil
				}

				return true, nil
			}

			ok1, err := inRange1()
			if err != nil {
				t.Fatal(err)
			}
			ok2, err := inRange2()
			if err != nil {
				t.Fatal(err)
			}

			if !ok1 && !ok2 {
				t.Fatalf("expected output to be in range")
			}
		})

		t.Run("Dropout(rate=0) / Forward(Full([10,10,100], 3)) with grad track / output equals Full([10,10,100], 3)", func(t *testing.T) {
			layer, err := layers.NewDropout(&layers.DropoutConfig{
				Rate: 0,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Full([]int{10, 10, 100}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Full([]int{10, 10, 100}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatalf("expected tensors to be equal")
			}
		})

		// ============================== validations ==============================

		t.Run("NewDropout(rate=-0.1) / returns error: Rate out of range [0,1)", func(t *testing.T) {
			_, err := layers.NewDropout(&layers.DropoutConfig{
				Rate: -0.1,
			})
			if err == nil {
				t.Fatalf("expected error because of non-positive 'Rate'")
			} else if err.Error() != "Dropout config data validation failed: expected 'Rate' to be in range [0,1): got (-0.100000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewDropout(rate=1) / returns error: Rate out of range [0,1)", func(t *testing.T) {
			_, err := layers.NewDropout(&layers.DropoutConfig{
				Rate: 1,
			})
			if err == nil {
				t.Fatalf("expected error because of 'Rate' being greater than or equal to (1)")
			} else if err.Error() != "Dropout config data validation failed: expected 'Rate' to be in range [0,1): got (1.000000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewDropout(rate=1.001) / returns error: Rate out of range [0,1)", func(t *testing.T) {
			_, err := layers.NewDropout(&layers.DropoutConfig{
				Rate: 1.001,
			})
			if err == nil {
				t.Fatalf("expected error because of 'Rate' being greater than or equal to (1)")
			} else if err.Error() != "Dropout config data validation failed: expected 'Rate' to be in range [0,1): got (1.001000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Dropout(nil) / Forward() with no input tensors / returns error: expected exactly one input tensor", func(t *testing.T) {
			layer, err := layers.NewDropout(nil)
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward()
			if err == nil {
				t.Fatalf("expected error because of not receiving one input tensor")
			} else if err.Error() != "Dropout input data validation failed: expected exactly one input tensor: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Dropout(nil) / Forward(x, x) with two input tensors / returns error: expected exactly one input tensor", func(t *testing.T) {
			layer, err := layers.NewDropout(nil)
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x, x)
			if err == nil {
				t.Fatalf("expected error because of not receiving one input tensor")
			} else if err.Error() != "Dropout input data validation failed: expected exactly one input tensor: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
