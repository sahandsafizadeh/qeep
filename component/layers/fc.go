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
	Inputs       int
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
		err = fmt.Errorf("FC config data validation failed: %w", err)
		return
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
		err = fmt.Errorf("FC input data validation failed: %w", err)
		return
	}

	y, err = c.forward(x)
	if err != nil {
		err = fmt.Errorf("FC forward failed: %w", err)
		return
	}

	return y, nil
}

func (c *FC) forward(x tensor.Tensor) (y tensor.Tensor, err error) {
	err = c.initWeights()
	if err != nil {
		return
	}

	w, err := c.Weight.UnSqueeze(1)
	if err != nil {
		return
	}

	b := c.Bias

	x, err = x.UnSqueeze(1) // last dim - 1
	if err != nil {
		return
	}

	y, err = w.MatMul(x)
	if err != nil {
		return
	}

	y, err = y.SumAlong(2) // last dim
	if err != nil {
		return
	}

	y, err = y.Add(b)
	if err != nil {
		return
	}

	return y, nil
}

func (c *FC) initWeights() (err error) {
	if c.Weight == nil {
		var w tensor.Tensor
		win := c.initializers[fcWeightKey]

		w, err = win.Init([]int{c.outputs}, c.device)
		if err != nil {
			return
		}

		err = c.validateInitializedWeight(w)
		if err != nil {
			err = fmt.Errorf("initialized 'Weight' validation failed: %w", err)
			return
		}

		c.Weight = w
	}

	if c.Bias == nil {
		var b tensor.Tensor
		bin := c.initializers[fcBiasKey]

		b, err = bin.Init([]int{c.outputs}, c.device)
		if err != nil {
			return
		}

		err = c.validateInitializedWeight(b)
		if err != nil {
			err = fmt.Errorf("initialized 'Bias' validation failed: %w", err)
			return
		}

		c.Bias = b
	}

	return nil
}

/* ----- helpers ----- */

func (c *FC) toValidInputs(xs []tensor.Tensor) (x tensor.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected exactly one input tensor: got (%d)", len(xs))
		return
	}

	x = xs[0]

	shape := x.Shape()

	if len(shape) != 2 {
		err = fmt.Errorf("expected input tensor to have exactly two dimensions (batch, data): got (%d)", len(shape))
		return
	}

	return x, nil
}

func (c *FC) validateInitializedWeight(t tensor.Tensor) (err error) {
	shape := t.Shape()

	if len(shape) != 1 {
		err = fmt.Errorf("expected initialized weights to have exactly one dimension")
		return
	}

	if shape[0] != c.outputs {
		err = fmt.Errorf("expected initialized size to match the output size: (%d) != (%d)", shape[0], c.outputs)
		return
	}

	return nil
}

func toValidFCConfig(iconf *FCConfig) (conf *FCConfig, err error) {
	if iconf == nil {
		err = fmt.Errorf("expected config not to be nil")
		return
	}

	conf = new(FCConfig)
	*conf = *iconf

	if conf.Inputs <= 0 {
		err = fmt.Errorf("expected 'Inputs' to be positive: got (%d)", conf.Inputs)
		return
	}

	if conf.Outputs <= 0 {
		err = fmt.Errorf("expected 'Outputs' to be positive: got (%d)", conf.Outputs)
		return
	}

	if conf.Initializers == nil {
		conf.Initializers = make(map[string]Initializer)
	}

	if _, ok := conf.Initializers[fcWeightKey]; !ok {
		conf.Initializers[fcWeightKey], err = initializers.NewXavierUniform(
			&initializers.XavierUniformConfig{
				FanIn:  conf.Inputs,
				FanOut: conf.Outputs,
			})
		if err != nil {
			return
		}
	}

	if _, ok := conf.Initializers[fcBiasKey]; !ok {
		conf.Initializers[fcBiasKey] = initializers.NewFull(
			&initializers.FullConfig{
				Value: 0.,
			})
		if err != nil {
			return
		}
	}

	if conf.Device == 0 {
		conf.Device = tensor.CPU
	}

	return conf, nil
}
