package layers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestFC(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("FC(1->1, weight=3, bias=1) / Forward([[-1]]) / returns [[-2]]", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Weight": initializers.NewFull(&initializers.FullConfig{Value: 3.}),
					"Bias":   initializers.NewFull(&initializers.FullConfig{Value: 1.}),
				},
				Device: dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{{-1.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{{-2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("FC(4->1, weight=-2, bias=3) / Forward([[-2,0,1,2]]) / returns [[1]]", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Inputs:  4,
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Weight": initializers.NewFull(&initializers.FullConfig{Value: -2.}),
					"Bias":   initializers.NewFull(&initializers.FullConfig{Value: 3.}),
				},
				Device: dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{{-2., 0., 1., 2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
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

		t.Run("FC(5->2, weight=5, bias=-1) / Forward([[-3,-2,1,1,5]]) / returns [[9,9]]", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Inputs:  5,
				Outputs: 2,
				Initializers: map[string]layers.Initializer{
					"Weight": initializers.NewFull(&initializers.FullConfig{Value: 5.}),
					"Bias":   initializers.NewFull(&initializers.FullConfig{Value: -1.}),
				},
				Device: dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{{-3., -2., 1., 1., 5.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{{9., 9.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("FC(8->3, weight=2, bias=1) / Forward(4x8 batch) / returns correct 4x3 output", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Inputs:  8,
				Outputs: 3,
				Initializers: map[string]layers.Initializer{
					"Weight": initializers.NewFull(&initializers.FullConfig{Value: 2.}),
					"Bias":   initializers.NewFull(&initializers.FullConfig{Value: 1.}),
				},
				Device: dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{
				{0., 1., 2., 3., 4., 5., 6., 7.},
				{1., 2., 3., 4., 5., 6., 7., 8.},
				{2., 3., 4., 5., 6., 7., 8., 9.},
				{3., 4., 5., 6., 7., 8., 9., 0.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{57., 57., 57.},
				{73., 73., 73.},
				{89., 89., 89.},
				{85., 85., 85.},
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

		t.Run("FC(8->3) / Weights() / returns 2 trainable weights pointing to Weight and Bias", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Inputs:  8,
				Outputs: 3,
				Initializers: map[string]layers.Initializer{
					"Weight": initializers.NewFull(&initializers.FullConfig{Value: 2.}),
					"Bias":   initializers.NewFull(&initializers.FullConfig{Value: 1.}),
				},
				Device: dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			weights := layer.Weights()
			if len(weights) != 2 {
				t.Fatalf("expected FC to have (2) trainable weights: got (%d)", len(weights))
			}

			expw := layers.Weight{
				Value:     &layer.Weight,
				Trainable: true,
			}
			if weights[0] != expw {
				t.Fatal("expected FC weight (0) to be trainable and point to 'Weight'")
			}

			expb := layers.Weight{
				Value:     &layer.Bias,
				Trainable: true,
			}
			if weights[1] != expb {
				t.Fatal("expected FC weight (1) to be trainable and point to 'Bias'")
			}
		})

		// ============================== validations ==============================

		t.Run("NewFC(nil) / returns error: nil config", func(t *testing.T) {
			_, err := layers.NewFC(nil)
			if err == nil {
				t.Fatal("expected error because of nil input config")
			} else if err.Error() != "FC config data validation failed: expected config not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewFC(Inputs=0) / returns error: non-positive Inputs", func(t *testing.T) {
			_, err := layers.NewFC(&layers.FCConfig{
				Inputs:  0,
				Outputs: 1,
			})
			if err == nil {
				t.Fatal("expected error because of non-positive 'Inputs'")
			} else if err.Error() != "FC config data validation failed: expected 'Inputs' to be positive: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewFC(Inputs=-1) / returns error: non-positive Inputs", func(t *testing.T) {
			_, err := layers.NewFC(&layers.FCConfig{
				Inputs:  -1,
				Outputs: 1,
			})
			if err == nil {
				t.Fatal("expected error because of non-positive 'Inputs'")
			} else if err.Error() != "FC config data validation failed: expected 'Inputs' to be positive: got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewFC(Outputs=0) / returns error: non-positive Outputs", func(t *testing.T) {
			_, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: 0,
			})
			if err == nil {
				t.Fatal("expected error because of non-positive 'Outputs'")
			} else if err.Error() != "FC config data validation failed: expected 'Outputs' to be positive: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewFC(Outputs=-1) / returns error: non-positive Outputs", func(t *testing.T) {
			_, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: -1,
			})
			if err == nil {
				t.Fatal("expected error because of non-positive 'Outputs'")
			} else if err.Error() != "FC config data validation failed: expected 'Outputs' to be positive: got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewFC with Weight initializer returning 0-D tensor / returns error: initialized weights must have exactly one dimension", func(t *testing.T) {
			_, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Weight": new(zeroDInitializer),
				},
			})
			if err == nil {
				t.Fatal("expected error because of weights initialized with more/less than one dimension")
			} else if err.Error() != "FC initialization failed: FC initialized weight validation failed: expected initialized weights to have exactly one dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewFC with Bias initializer returning 0-D tensor / returns error: initialized weights must have exactly one dimension", func(t *testing.T) {
			_, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Bias": new(zeroDInitializer),
				},
			})
			if err == nil {
				t.Fatal("expected error because of weights initialized with more/less than one dimension")
			} else if err.Error() != "FC initialization failed: FC initialized weight validation failed: expected initialized weights to have exactly one dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewFC with Weight initializer returning 2-D tensor / returns error: initialized weights must have exactly one dimension", func(t *testing.T) {
			_, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Weight": new(twoDInitializer),
				},
			})
			if err == nil {
				t.Fatal("expected error because of weights initialized with more/less than one dimension")
			} else if err.Error() != "FC initialization failed: FC initialized weight validation failed: expected initialized weights to have exactly one dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewFC with Bias initializer returning 2-D tensor / returns error: initialized weights must have exactly one dimension", func(t *testing.T) {
			_, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Bias": new(twoDInitializer),
				},
			})
			if err == nil {
				t.Fatal("expected error because of weights initialized with more/less than one dimension")
			} else if err.Error() != "FC initialization failed: FC initialized weight validation failed: expected initialized weights to have exactly one dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewFC with Weight initializer returning wrong 1-D size / returns error: Weight size must match Outputs", func(t *testing.T) {
			_, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Weight": new(wrong1DInitializer),
				},
			})
			if err == nil {
				t.Fatal("expected error because of 'Weight' being initialized with mismatched size")
			} else if err.Error() != "FC initialization failed: FC initialized weight validation failed: expected initialized 'Weight' size to match 'Outputs': (2) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewFC with Bias initializer returning wrong 1-D size / returns error: Bias size must match Outputs", func(t *testing.T) {
			_, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Bias": new(wrong1DInitializer),
				},
			})
			if err == nil {
				t.Fatal("expected error because of 'Bias' being initialized with mismatched size")
			} else if err.Error() != "FC initialization failed: FC initialized weight validation failed: expected initialized 'Bias' size to match 'Outputs': (2) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("FC(1->1) / Forward() with 0 input tensors / returns error: expected exactly one input tensor", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: 1,
				Device:  dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward()
			if err == nil {
				t.Fatal("expected error because of not receiving one input tensor")
			} else if err.Error() != "FC input data validation failed: expected exactly one input tensor: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("FC(1->1) / Forward(x, x) with 2 input tensors / returns error: expected exactly one input tensor", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: 1,
				Device:  dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x, x)
			if err == nil {
				t.Fatal("expected error because of not receiving one input tensor")
			} else if err.Error() != "FC input data validation failed: expected exactly one input tensor: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("FC(1->1) / Forward(1-D tensor) / returns error: input must have exactly two dimensions", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: 1,
				Device:  dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros([]int{1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x)
			if err == nil {
				t.Fatal("expected error because of tensor having more/less than two dimensions")
			} else if err.Error() != "FC input data validation failed: expected input tensor to have exactly two dimensions (batch, data): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("FC(1->1) / Forward(3-D tensor) / returns error: input must have exactly two dimensions", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: 1,
				Device:  dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros([]int{1, 1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x)
			if err == nil {
				t.Fatal("expected error because of tensor having more/less than two dimensions")
			} else if err.Error() != "FC input data validation failed: expected input tensor to have exactly two dimensions (batch, data): got (3)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

/* ----- helpers ----- */

type zeroDInitializer struct{}
type twoDInitializer struct{}
type wrong1DInitializer struct{}

func (c *zeroDInitializer) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	return tensor.Zeros(nil, nil)
}

func (c *twoDInitializer) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	return tensor.Zeros([]int{1, 1}, nil)
}

func (c *wrong1DInitializer) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	return tensor.Zeros([]int{shape[0] + 1}, nil)
}
