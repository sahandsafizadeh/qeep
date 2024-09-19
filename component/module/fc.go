package component

import (
	"fmt"

	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type FC struct {
	FCConfig
	w qt.Tensor
	b qt.Tensor
}

type FCConfig struct {
	N int32
}

func NewFC(conf *FCConfig) (c *FC, err error) {
	conf, err = toValidFCConfig(conf)
	if err != nil {
		return
	}

	return &FC{
		FCConfig: *conf,
	}, nil
}

func (c *FC) Forward(xs ...qt.Tensor) (y qt.Tensor, err error) {
	x, err := toValidFCInputs(xs)
	if err != nil {
		return
	}

	return c.forward(x)
}

func (c *FC) Weights() []*qt.Tensor {
	return []*qt.Tensor{&c.w, &c.b}
}

func (c *FC) forward(x qt.Tensor) (y qt.Tensor, err error) {
	w, err := c.w.UnSqueeze(1)
	if err != nil {
		return
	}

	// last dim - 1
	x, err = x.UnSqueeze(1)
	if err != nil {
		return
	}

	y, err = w.MatMul(x)
	if err != nil {
		return
	}

	// last dim
	y, err = y.SumAlong(2)
	if err != nil {
		return
	}

	y, err = y.Add(c.b)
	if err != nil {
		return
	}

	return y, nil
}

/* ----- helpers ----- */

func toValidFCConfig(iconf *FCConfig) (conf *FCConfig, err error) {
	if iconf == nil {
		err = fmt.Errorf("expected fc configuration not to be nil")
		return
	}

	conf = new(FCConfig)
	*conf = *iconf

	if conf.N <= 0 {
		err = fmt.Errorf("expected fc 'N' to be positive: got (%d)", conf.N)
		return
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
