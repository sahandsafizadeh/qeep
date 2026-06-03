package activations

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Softmax struct {
	dim int
}

type SoftmaxConfig struct {
	Dim int
}

func NewSoftmax(conf *SoftmaxConfig) (c *Softmax, err error) {
	conf, err = toValidSoftmaxConfig(conf)
	if err != nil {
		return c, fmt.Errorf("Softmax config data validation failed: %w", err)
	}

	return &Softmax{
		dim: conf.Dim,
	}, nil
}

func (c *Softmax) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	x, err := c.toValidInputs(xs)
	if err != nil {
		return y, fmt.Errorf("Softmax input data validation failed: %w", err)
	}

	y, err = c.forward(x)
	if err != nil {
		return y, fmt.Errorf("Softmax forward failed: %w", err)
	}

	return y, nil
}

func (c *Softmax) forward(x tensor.Tensor) (y tensor.Tensor, err error) {
	x = x.Exp()

	s, err := x.SumAlong(c.dim)
	if err != nil {
		return y, err
	}

	s, err = s.UnSqueeze(c.dim)
	if err != nil {
		return y, err
	}

	return x.Div(s)
}

/* ----- helpers ----- */

func (c *Softmax) toValidInputs(xs []tensor.Tensor) (x tensor.Tensor, err error) {
	if len(xs) != 1 {
		return x, fmt.Errorf("expected exactly one input tensor: got (%d)", len(xs))
	}

	x = xs[0]

	shape := x.Shape()

	if len(shape) <= c.dim {
		return x, fmt.Errorf("expected input tensor shape to match 'Dim': %v !~ (%d)", shape, c.dim)
	}

	return x, nil
}

func toValidSoftmaxConfig(iconf *SoftmaxConfig) (conf *SoftmaxConfig, err error) {
	if iconf == nil {
		return conf, fmt.Errorf("expected config not to be nil")
	}

	conf = new(SoftmaxConfig)
	*conf = *iconf

	if conf.Dim < 0 {
		return conf, fmt.Errorf("expected 'Dim' not to be negative: got (%d)", conf.Dim)
	}

	return conf, nil
}
