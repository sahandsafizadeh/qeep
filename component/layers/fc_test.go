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

		t.Run("FC(1->2, seq weight/bias) / Forward([[3]]) / returns [[4, 8]]", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Outputs: 2,
				Initializers: map[string]layers.Initializer{
					"Weight": new(seqInitializer),
					"Bias":   new(seqInitializer),
				},
				Device: dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			/*
				weight = [[1, 2]]
				bias = [1, 2]
			*/

			x, err := tensor.Of([][]float64{{3.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{{4., 8.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("FC(2->3, seq weight/bias) / Forward([[1, 2]]) / returns [[10, 14, 18]]", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Outputs: 3,
				Initializers: map[string]layers.Initializer{
					"Weight": new(seqInitializer),
					"Bias":   new(seqInitializer),
				},
				Device: dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			/*
				weight = [
							[1,2,3],
							[4,5,6],
						 ]
				bias = [1, 2]
			*/

			x, err := tensor.Of([][]float64{{1., 2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{{10., 14., 18.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("FC(8->3, seq weight/bias) / Forward(5x8 batch) / returns correct 5x3 output", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Outputs: 3,
				Initializers: map[string]layers.Initializer{
					"Weight": new(seqInitializer),
					"Bias":   new(seqInitializer),
				},
				Device: dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			/*
				weight = [
							[1,2,3],
							[4,5,6],
							[7,8,9],
							[10,11,12],
							[13,14,15],
							[16,17,18],
							[19,20,21],
							[22,23,24],
						  ]
				bias = [1, 2, 3]
			*/

			x, err := tensor.Of([][]float64{
				{1., 1., 1., 1., 1., 1., 1., 1.},
				{2., 2., 2., 2., 2., 2., 2., 2.},
				{0., 0., 0., 0., 0., 0., 0., 0.},
				{1., 0., 0., 0., 0., 0., 0., 0.},
				{0., 0., 0., 0., 0., 0., 0., 1.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			exp, err := tensor.Of([][]float64{
				{93., 102., 111.},
				{185., 202., 219.},
				{1., 2., 3.},
				{2., 4., 6.},
				{23., 25., 27.},
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

		t.Run("FC layer / Weights() / before first forward, both weights are uninitialized", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Outputs: 3,
				Device:  dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			weights := layer.Weights()
			if len(weights) != 2 {
				t.Fatalf("expected FC to have (2) weights: got (%d)", len(weights))
			}

			if !weights[0].Trainable || *weights[0].Value != nil {
				t.Fatal("expected FC weight (0) to be trainable with nil value")
			}
			if !weights[1].Trainable || *weights[1].Value != nil {
				t.Fatal("expected FC weight (1) to be trainable with nil value")
			}
		})

		t.Run("FC layer / Weights() / after Forward(), returns 2 trainable weights pointing to Weight and Bias", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Outputs: 3,
				Device:  dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Ones([]int{4, 8}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			weights := layer.Weights()
			if len(weights) != 2 {
				t.Fatalf("expected FC to have (2) weights: got (%d)", len(weights))
			}

			if !weights[0].Trainable || *weights[0].Value == nil || weights[0].Value != &layer.Weight {
				t.Fatal("expected FC weight (0) to be trainable, non-nil and point to 'Weight'")
			}
			if !weights[1].Trainable || *weights[1].Value == nil || weights[1].Value != &layer.Bias {
				t.Fatal("expected FC weight (1) to be trainable, non-nil and point to 'Bias'")
			}
		})

		t.Run("FC layer / Weights() / pre-initialized weights stay the same after Forward()", func(t *testing.T) {
			// ----- given -----
			layer, err := layers.NewFC(&layers.FCConfig{
				Outputs: 3,
				Device:  dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			w, err := tensor.Ones([]int{8, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Ones([]int{3}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			layer.Weight = w
			layer.Bias = b

			// ----- when -----
			x, err := tensor.Ones([]int{4, 8}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			weights := layer.Weights()
			if len(weights) != 2 {
				t.Fatalf("expected FC to have (2) weights: got (%d)", len(weights))
			}

			if !weights[0].Trainable || *weights[0].Value != w || weights[0].Value != &layer.Weight {
				t.Fatal("expected FC weight (0) to be trainable, stay the same and point to 'Weight'")
			}
			if !weights[1].Trainable || *weights[1].Value != b || weights[1].Value != &layer.Bias {
				t.Fatal("expected FC weight (1) to be trainable, stay the same and point to 'Bias'")
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

		t.Run("NewFC(Outputs=0) / returns error: non-positive Outputs", func(t *testing.T) {
			_, err := layers.NewFC(&layers.FCConfig{Outputs: 0})
			if err == nil {
				t.Fatal("expected error because of non-positive 'Outputs'")
			} else if err.Error() != "FC config data validation failed: expected 'Outputs' to be positive: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewFC(Outputs=-1) / returns error: non-positive Outputs", func(t *testing.T) {
			_, err := layers.NewFC(&layers.FCConfig{Outputs: -1})
			if err == nil {
				t.Fatal("expected error because of non-positive 'Outputs'")
			} else if err.Error() != "FC config data validation failed: expected 'Outputs' to be positive: got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("FC(1->1) / Forward() with 0 input tensors / returns error: expected exactly one input tensor", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
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

		t.Run("FC with Weight initializer returning 1-D tensor / Forward() / returns error: initialized weight must have exactly two dimensions", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Weight": new(oneDInitializer),
				},
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x)
			if err == nil {
				t.Fatal("expected error because of weight initialized with wrong number of dimensions")
			} else if err.Error() != "FC weight initialization failed: expected 'Weight' parameter to have exactly two dimensions" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("FC with Weight initializer returning 3-D tensor / Forward() / returns error: initialized weight must have exactly two dimensions", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Weight": new(threeDInitializer),
				},
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x)
			if err == nil {
				t.Fatal("expected error because of weight initialized with wrong number of dimensions")
			} else if err.Error() != "FC weight initialization failed: expected 'Weight' parameter to have exactly two dimensions" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("FC with Bias initializer returning 0-D tensor / Forward() / returns error: initialized bias must have exactly one dimension", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Bias": new(zeroDInitializer),
				},
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x)
			if err == nil {
				t.Fatal("expected error because of weights initialized with more/less than one dimension")
			} else if err.Error() != "FC weight initialization failed: expected 'Bias' parameter to have exactly one dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("FC with Bias initializer returning 2-D tensor / Forward() / returns error: initialized bias must have exactly one dimension", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Bias": new(twoDInitializer),
				},
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x)
			if err == nil {
				t.Fatal("expected error because of weights initialized with more/less than one dimension")
			} else if err.Error() != "FC weight initialization failed: expected 'Bias' parameter to have exactly one dimension" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("FC with Weight initializer returning wrong 2-D row size / Forward() / returns error: Weight row size must match input size", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Weight": new(wrong2DInitializer),
				},
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x)
			if err == nil {
				t.Fatal("expected error because of weight initialized with wrong row size")
			} else if err.Error() != "FC weight initialization failed: expected 'Weight' parameter row size to match the input size: (2) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("FC with Bias initializer returning wrong 1-D size / Forward() / returns error: Bias size must match Outputs", func(t *testing.T) {
			layer, err := layers.NewFC(&layers.FCConfig{
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Bias": new(wrong1DInitializer),
				},
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros([]int{1, 1}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x)
			if err == nil {
				t.Fatal("expected error because of weights initialized with wrong size")
			} else if err.Error() != "FC weight initialization failed: expected 'Bias' parameter size to match the output size: (2) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

/* ----- helpers ----- */

type seqInitializer struct{}
type zeroDInitializer struct{}
type oneDInitializer struct{}
type twoDInitializer struct{}
type threeDInitializer struct{}
type wrong1DInitializer struct{}
type wrong2DInitializer struct{}

func (c *seqInitializer) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	switch len(shape) {
	case 1:
		v := 1.
		data := make([]float64, shape[0])
		for i := range data {
			data[i] = v
			v++
		}

		return tensor.Of(data, &tensor.Config{Device: device})

	case 2:
		v := 1.
		data := make([][]float64, shape[0])
		for i := range data {
			data[i] = make([]float64, shape[1])
			for j := range data[i] {
				data[i][j] = v
				v++
			}
		}

		return tensor.Of(data, &tensor.Config{Device: device})

	default:
		panic("unsupported sequence initializer")
	}
}

func (c *zeroDInitializer) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	return tensor.Zeros(nil, nil)
}

func (c *oneDInitializer) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	return tensor.Zeros([]int{1}, nil)
}

func (c *twoDInitializer) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	return tensor.Zeros([]int{1, 1}, nil)
}

func (c *threeDInitializer) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	return tensor.Zeros([]int{1, 1, 1}, nil)
}

func (c *wrong1DInitializer) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	return tensor.Zeros([]int{shape[0] + 1}, nil)
}

func (c *wrong2DInitializer) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	return tensor.Zeros([]int{shape[0] + 1, shape[1]}, nil)
}
