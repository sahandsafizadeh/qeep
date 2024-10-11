package weighted

import (
	"fmt"

	qc "github.com/sahandsafizadeh/qeep/component"
	qci "github.com/sahandsafizadeh/qeep/component/initializer"
	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type FC struct {
	Weight qt.Tensor
	Bias   qt.Tensor
}

type FCConfig struct {
	Inputs       int
	Outputs      int
	Initializers map[string]qc.Initializer
}

const (
	fcWeightKey = "Weight"
	fcBiasKey   = "Bias"
)

func NewFC(conf *FCConfig) (c *FC, err error) {
	conf, err = toValidFCConfig(conf)
	if err != nil {
		return
	}

	var (
		win = conf.Initializers[fcWeightKey]
		bin = conf.Initializers[fcBiasKey]
	)

	w, err := win.Init([]int{conf.Outputs})
	if err != nil {
		return
	}

	b, err := bin.Init([]int{conf.Outputs})
	if err != nil {
		return
	}

	return &FC{
		Weight: w,
		Bias:   b,
	}, nil
}

func (c *FC) TrainableWeights() []*qt.Tensor {
	return []*qt.Tensor{&c.Weight, &c.Bias}
}

func (c *FC) Forward(xs ...qt.Tensor) (y qt.Tensor, err error) {
	x, err := toValidFCInputs(xs)
	if err != nil {
		return
	}

	return c.forward(x)
}

func (c *FC) forward(x qt.Tensor) (y qt.Tensor, err error) {
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

func toValidFCConfig(iconf *FCConfig) (conf *FCConfig, err error) {
	if iconf == nil {
		err = fmt.Errorf("expected fc config not to be nil")
		return
	}

	conf = new(FCConfig)
	*conf = *iconf

	if conf.Inputs <= 0 {
		err = fmt.Errorf("expected fc 'Inputs' to be positive: got (%d)", conf.Inputs)
		return
	}

	if conf.Outputs <= 0 {
		err = fmt.Errorf("expected fc 'Outputs' to be positive: got (%d)", conf.Outputs)
		return
	}

	if conf.Initializers == nil {
		conf.Initializers = make(map[string]qc.Initializer)
	}

	if _, ok := conf.Initializers[fcWeightKey]; !ok {
		conf.Initializers[fcWeightKey], err = qci.NewXavierUniform(
			&qci.XavierUniformConfig{
				FanIn:  conf.Inputs,
				FanOut: conf.Outputs,
			})
		if err != nil {
			return
		}
	}

	if _, ok := conf.Initializers[fcBiasKey]; !ok {
		conf.Initializers[fcBiasKey] = qci.NewFull(
			&qci.FullConfig{
				Value: 0.,
			})
		if err != nil {
			return
		}
	}

	return conf, nil
}

func toValidFCInputs(xs []qt.Tensor) (x qt.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected fc to receive exactly one input: got (%d)", len(xs))
		return
	}

	return xs[0], nil
}
