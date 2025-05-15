package layers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestFC(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		layer, err := layers.NewFC(&layers.FCConfig{
			Inputs:  1,
			Outputs: 1,
			Initializers: map[string]layers.Initializer{
				"Weight": initializers.NewFull(&initializers.FullConfig{Value: 3.}),
				"Bias":   initializers.NewFull(&initializers.FullConfig{Value: 1.}),
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err := tensor.Of([][]float64{{-1.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err := layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.Of([][]float64{{-2.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		layer, err = layers.NewFC(&layers.FCConfig{
			Inputs:  4,
			Outputs: 1,
			Initializers: map[string]layers.Initializer{
				"Weight": initializers.NewFull(&initializers.FullConfig{Value: -2.}),
				"Bias":   initializers.NewFull(&initializers.FullConfig{Value: 3.}),
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err = tensor.Of([][]float64{{-2., 0., 1., 2.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][]float64{{1.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		layer, err = layers.NewFC(&layers.FCConfig{
			Inputs:  5,
			Outputs: 2,
			Initializers: map[string]layers.Initializer{
				"Weight": initializers.NewFull(&initializers.FullConfig{Value: 5.}),
				"Bias":   initializers.NewFull(&initializers.FullConfig{Value: -1.}),
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err = tensor.Of([][]float64{{-3., -2., 1., 1., 5.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][]float64{{9., 9.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		layer, err = layers.NewFC(&layers.FCConfig{
			Inputs:  8,
			Outputs: 3,
			Initializers: map[string]layers.Initializer{
				"Weight": initializers.NewFull(&initializers.FullConfig{Value: 2.}),
				"Bias":   initializers.NewFull(&initializers.FullConfig{Value: 1.}),
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err = tensor.Of([][]float64{
			{0., 1., 2., 3., 4., 5., 6., 7.},
			{1., 2., 3., 4., 5., 6., 7., 8.},
			{2., 3., 4., 5., 6., 7., 8., 9.},
			{3., 4., 5., 6., 7., 8., 9., 0.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.Of([][]float64{
			{57., 57., 57.},
			{73., 73., 73.},
			{89., 89., 89.},
			{85., 85., 85.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

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

		/* ------------------------------ */

	})
}

func TestValidationFC(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		_, err := layers.NewFC(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input config")
		} else if err.Error() != "FC config data validation failed: expected config not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layers.NewFC(&layers.FCConfig{
			Inputs:  0,
			Outputs: 1,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'Inputs'")
		} else if err.Error() != "FC config data validation failed: expected 'Inputs' to be positive: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layers.NewFC(&layers.FCConfig{
			Inputs:  -1,
			Outputs: 1,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'Inputs'")
		} else if err.Error() != "FC config data validation failed: expected 'Inputs' to be positive: got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layers.NewFC(&layers.FCConfig{
			Inputs:  1,
			Outputs: 0,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'Outputs'")
		} else if err.Error() != "FC config data validation failed: expected 'Outputs' to be positive: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layers.NewFC(&layers.FCConfig{
			Inputs:  1,
			Outputs: -1,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'Outputs'")
		} else if err.Error() != "FC config data validation failed: expected 'Outputs' to be positive: got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		_, err = layers.NewFC(&layers.FCConfig{
			Inputs:  1,
			Outputs: 1,
			Initializers: map[string]layers.Initializer{
				"Weight": new(zeroDInitializer),
			},
		})
		if err == nil {
			t.Fatalf("expected error because of weights initialized with more/less than one dimension")
		} else if err.Error() != "FC initialized weight validation failed: expected initialized weights to have exactly one dimension" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layers.NewFC(&layers.FCConfig{
			Inputs:  1,
			Outputs: 1,
			Initializers: map[string]layers.Initializer{
				"Bias": new(zeroDInitializer),
			},
		})
		if err == nil {
			t.Fatalf("expected error because of weights initialized with more/less than one dimension")
		} else if err.Error() != "FC initialized weight validation failed: expected initialized weights to have exactly one dimension" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layers.NewFC(&layers.FCConfig{
			Inputs:  1,
			Outputs: 1,
			Initializers: map[string]layers.Initializer{
				"Weight": new(twoDInitializer),
			},
		})
		if err == nil {
			t.Fatalf("expected error because of weights initialized with more/less than one dimension")
		} else if err.Error() != "FC initialized weight validation failed: expected initialized weights to have exactly one dimension" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layers.NewFC(&layers.FCConfig{
			Inputs:  1,
			Outputs: 1,
			Initializers: map[string]layers.Initializer{
				"Bias": new(twoDInitializer),
			},
		})
		if err == nil {
			t.Fatalf("expected error because of weights initialized with more/less than one dimension")
		} else if err.Error() != "FC initialized weight validation failed: expected initialized weights to have exactly one dimension" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layers.NewFC(&layers.FCConfig{
			Inputs:  1,
			Outputs: 1,
			Initializers: map[string]layers.Initializer{
				"Weight": new(wrong1DInitializer),
			},
		})
		if err == nil {
			t.Fatalf("expected error because of 'Weight' being initialized with mismatched size")
		} else if err.Error() != "FC initialized weight validation failed: expected initialized 'Weight' size to match 'Outputs': (2) != (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layers.NewFC(&layers.FCConfig{
			Inputs:  1,
			Outputs: 1,
			Initializers: map[string]layers.Initializer{
				"Bias": new(wrong1DInitializer),
			},
		})
		if err == nil {
			t.Fatalf("expected error because of 'Bias' being initialized with mismatched size")
		} else if err.Error() != "FC initialized weight validation failed: expected initialized 'Bias' size to match 'Outputs': (2) != (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		layer, err := layers.NewFC(&layers.FCConfig{
			Inputs:  1,
			Outputs: 1,
		})
		if err != nil {
			t.Fatal(err)
		}

		x1, err := tensor.Zeros([]int{1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		x2, err := tensor.Zeros([]int{1, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		x3, err := tensor.Zeros([]int{1, 1, 1}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = layer.Forward()
		if err == nil {
			t.Fatalf("expected error because of not receiving one input tensor")
		} else if err.Error() != "FC input data validation failed: expected exactly one input tensor: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layer.Forward(x2, x2)
		if err == nil {
			t.Fatalf("expected error because of not receiving one input tensor")
		} else if err.Error() != "FC input data validation failed: expected exactly one input tensor: got (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layer.Forward(x1)
		if err == nil {
			t.Fatalf("expected error because of tensor having more/less than two dimensions")
		} else if err.Error() != "FC input data validation failed: expected input tensor to have exactly two dimensions (batch, data): got (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layer.Forward(x3)
		if err == nil {
			t.Fatalf("expected error because of tensor having more/less than two dimensions")
		} else if err.Error() != "FC input data validation failed: expected input tensor to have exactly two dimensions (batch, data): got (3)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

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
