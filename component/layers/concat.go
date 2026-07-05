package layers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Concat struct {
	dim int
}

type ConcatConfig struct {
	Dim int
}

func NewConcat(conf *ConcatConfig) (c *Concat, err error) {
	conf, err = toValidConcatConfig(conf)
	if err != nil {
		return c, fmt.Errorf("Concat config data validation failed: %w", err)
	}

	return &Concat{
		dim: conf.Dim,
	}, nil
}

func (c *Concat) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	xs, err = c.toValidInputs(xs)
	if err != nil {
		return y, fmt.Errorf("Concat input data validation failed: %w", err)
	}

	y, err = c.forward(xs)
	if err != nil {
		return y, fmt.Errorf("Concat forward failed: %w", err)
	}

	return y, nil
}

func (c *Concat) forward(xs []tensor.Tensor) (y tensor.Tensor, err error) {
	return tensor.Concat(xs, c.dim)
}

/* ----- helpers ----- */

func (c *Concat) toValidInputs(xs []tensor.Tensor) ([]tensor.Tensor, error) {
	if len(xs) < 2 {
		return xs, fmt.Errorf("expected at least two input tensors: got (%d)", len(xs))
	}

	return xs, nil
}

func toValidConcatConfig(iconf *ConcatConfig) (conf *ConcatConfig, err error) {
	if iconf == nil {
		return conf, fmt.Errorf("expected config not to be nil")
	}

	conf = new(ConcatConfig)
	*conf = *iconf

	if conf.Dim < 0 {
		return conf, fmt.Errorf("expected 'Dim' not to be negative: got (%d)", conf.Dim)
	}

	return conf, nil
}
