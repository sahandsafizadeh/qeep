package layers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

// FC is a fully connected (dense) layer that computes y = x·W + b.
type FC struct {
	Weight tensor.Tensor
	Bias   tensor.Tensor

	outputs      int
	initializers map[string]Initializer
	device       tensor.Device
}

type FCConfig struct {
	Outputs      int
	Initializers map[string]Initializer
	Device       tensor.Device
}

const (
	fcWeightKey = "Weight"
	fcBiasKey   = "Bias"
)

func NewFC(conf *FCConfig) (c *FC, err error) {
	conf, err = toValidFCConfig(conf)
	if err != nil {
		return c, fmt.Errorf("FC config data validation failed: %w", err)
	}

	return &FC{
		outputs:      conf.Outputs,
		initializers: conf.Initializers,
		device:       conf.Device,
	}, nil
}

func (c *FC) Weights() []Weight {
	return []Weight{
		{
			Value:     &c.Weight,
			Trainable: true,
		},
		{
			Value:     &c.Bias,
			Trainable: true,
		},
	}
}

func (c *FC) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	x, err := c.toValidInputs(xs)
	if err != nil {
		return y, fmt.Errorf("FC input data validation failed: %w", err)
	}

	err = c.initWeights(x)
	if err != nil {
		return y, fmt.Errorf("FC weight initialization failed: %w", err)
	}

	y, err = c.forward(x)
	if err != nil {
		return y, fmt.Errorf("FC forward failed: %w", err)
	}

	return y, nil
}

func (c *FC) forward(x tensor.Tensor) (y tensor.Tensor, err error) {
	w, err := c.Weight.UnSqueeze(1)
	if err != nil {
		return y, err
	}

	b := c.Bias

	x, err = x.UnSqueeze(1) // last dim - 1
	if err != nil {
		return y, err
	}

	y, err = w.MatMul(x)
	if err != nil {
		return y, err
	}

	y, err = y.SumAlong(2) // last dim
	if err != nil {
		return y, err
	}

	y, err = y.Add(b)
	if err != nil {
		return y, err
	}

	return y, nil
}

func (c *FC) initWeights(x tensor.Tensor) (err error) {
	if c.Weight == nil {
		initializer, ok := c.initializers[fcWeightKey]
		if !ok {
			initializer, err = initializers.NewXavierUniform(&initializers.XavierUniformConfig{
				FanIn:  x.Shape()[1],
				FanOut: c.outputs,
			})
			if err != nil {
				return err
			}
		}

		w, err := initializer.Init([]int{c.outputs}, c.device)
		if err != nil {
			return err
		}

		err = c.validateInitializedWeight(w)
		if err != nil {
			return err
		}

		c.Weight = w
	}

	if c.Bias == nil {
		initializer, ok := c.initializers[fcBiasKey]
		if !ok {
			initializer = initializers.NewFull(&initializers.FullConfig{
				Value: 0.,
			})
		}

		b, err := initializer.Init([]int{c.outputs}, c.device)
		if err != nil {
			return err
		}

		err = c.validateInitializedWeight(b)
		if err != nil {
			return err
		}

		c.Bias = b
	}

	return nil
}

/* ----- helpers ----- */

func (c *FC) toValidInputs(xs []tensor.Tensor) (x tensor.Tensor, err error) {
	if len(xs) != 1 {
		return x, fmt.Errorf("expected exactly one input tensor: got (%d)", len(xs))
	}

	x = xs[0]

	shape := x.Shape()

	if len(shape) != 2 {
		return x, fmt.Errorf("expected input tensor to have exactly two dimensions (batch, data): got (%d)", len(shape))
	}

	return x, nil
}

func (c *FC) validateInitializedWeight(t tensor.Tensor) (err error) {
	shape := t.Shape()

	if len(shape) != 1 {
		return fmt.Errorf("expected initialized weights to have exactly one dimension")
	}

	if shape[0] != c.outputs {
		return fmt.Errorf("expected initialized size to match the output size: (%d) != (%d)", shape[0], c.outputs)
	}

	return nil
}

func toValidFCConfig(iconf *FCConfig) (conf *FCConfig, err error) {
	if iconf == nil {
		return conf, fmt.Errorf("expected config not to be nil")
	}

	conf = new(FCConfig)
	*conf = *iconf

	if conf.Device == 0 {
		conf.Device = tensor.CPU
	}

	if conf.Outputs <= 0 {
		return conf, fmt.Errorf("expected 'Outputs' to be positive: got (%d)", conf.Outputs)
	}

	return conf, nil
}
