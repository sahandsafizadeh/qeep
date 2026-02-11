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

// FCConfig specifies the layer dimensions, target device, and weight initializers.
// Initializers map uses keys "Weight" and "Bias"; defaults are applied if not provided.
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

// NewFC creates a fully connected layer with the given configuration.
// Returns an error if config validation fails.
func NewFC(conf *FCConfig) (c *FC, err error) {
	conf, err = toValidFCConfig(conf)
	if err != nil {
		err = fmt.Errorf("FC config data validation failed: %w", err)
		return
	}

	var (
		win = conf.Initializers[fcWeightKey]
		bin = conf.Initializers[fcBiasKey]
	)

	w, err := win.Init([]int{conf.Outputs}, conf.Device)
	if err != nil {
		return
	}

	b, err := bin.Init([]int{conf.Outputs}, conf.Device)
	if err != nil {
		return
	}

	err = validateInitializedWeights(w, b, conf)
	if err != nil {
		err = fmt.Errorf("FC initialized weight validation failed: %w", err)
		return
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
		err = fmt.Errorf("FC input data validation failed: %w", err)
		return
	}

	return c.forward(x)
}

func (c *FC) forward(x tensor.Tensor) (y tensor.Tensor, err error) {
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

func validateInitializedWeights(w tensor.Tensor, b tensor.Tensor, conf *FCConfig) (err error) {
	shapew := w.Shape()
	shapeb := b.Shape()

	if len(shapew) != 1 || len(shapeb) != 1 {
		err = fmt.Errorf("expected initialized weights to have exactly one dimension")
		return
	}

	if shapew[0] != conf.Outputs {
		err = fmt.Errorf("expected initialized 'Weight' size to match 'Outputs': (%d) != (%d)", shapew[0], conf.Outputs)
		return
	}

	if shapeb[0] != conf.Outputs {
		err = fmt.Errorf("expected initialized 'Bias' size to match 'Outputs': (%d) != (%d)", shapeb[0], conf.Outputs)
		return
	}

	return nil
}
