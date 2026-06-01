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
		return c, fmt.Errorf("FC config data validation failed: %w", err)
	}

	c, err = newFC(conf)
	if err != nil {
		return c, fmt.Errorf("FC initialization failed: %w", err)
	}

	return c, nil
}

func newFC(conf *FCConfig) (c *FC, err error) {
	var (
		win = conf.Initializers[fcWeightKey]
		bin = conf.Initializers[fcBiasKey]
	)

	w, err := win.Init([]int{conf.Outputs}, conf.Device)
	if err != nil {
		return c, err
	}

	b, err := bin.Init([]int{conf.Outputs}, conf.Device)
	if err != nil {
		return c, err
	}

	err = validateInitializedWeights(w, b, conf)
	if err != nil {
		return c, fmt.Errorf("FC initialized weight validation failed: %w", err)
	}

	return &FC{
		Weight: w,
		Bias:   b,
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

func toValidFCConfig(iconf *FCConfig) (conf *FCConfig, err error) {
	if iconf == nil {
		return conf, fmt.Errorf("expected config not to be nil")
	}

	conf = new(FCConfig)
	*conf = *iconf

	if conf.Inputs <= 0 {
		return conf, fmt.Errorf("expected 'Inputs' to be positive: got (%d)", conf.Inputs)
	}

	if conf.Outputs <= 0 {
		return conf, fmt.Errorf("expected 'Outputs' to be positive: got (%d)", conf.Outputs)
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
			return conf, err
		}
	}

	if _, ok := conf.Initializers[fcBiasKey]; !ok {
		conf.Initializers[fcBiasKey] = initializers.NewFull(
			&initializers.FullConfig{
				Value: 0.,
			})
	}

	if conf.Device == 0 {
		conf.Device = tensor.CPU
	}

	return conf, nil
}

func validateInitializedWeights(w tensor.Tensor, b tensor.Tensor, conf *FCConfig) (err error) {
	shapew := w.Shape()
	shapeb := b.Shape()

	if len(shapew) != 1 || len(shapeb) != 1 {
		return fmt.Errorf("expected initialized weights to have exactly one dimension")
	}

	if shapew[0] != conf.Outputs {
		return fmt.Errorf("expected initialized 'Weight' size to match 'Outputs': (%d) != (%d)", shapew[0], conf.Outputs)
	}

	if shapeb[0] != conf.Outputs {
		return fmt.Errorf("expected initialized 'Bias' size to match 'Outputs': (%d) != (%d)", shapeb[0], conf.Outputs)
	}

	return nil
}
