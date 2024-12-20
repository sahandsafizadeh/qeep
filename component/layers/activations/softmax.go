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

const softmaxDefaultDim = 0

func NewSoftmax(conf *SoftmaxConfig) (c *Softmax, err error) {
	conf, err = toValidSoftmaxConfig(conf)
	if err != nil {
		err = fmt.Errorf("Softmax config data validation failed: %w", err)
		return
	}

	return &Softmax{
		dim: conf.Dim,
	}, nil
}

func (c *Softmax) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	x, err := c.toValidInputs(xs)
	if err != nil {
		err = fmt.Errorf("Softmax input data validation failed: %w", err)
		return
	}

	return c.forward(x)
}

func (c *Softmax) forward(x tensor.Tensor) (y tensor.Tensor, err error) {
	x = x.Exp()

	s, err := x.SumAlong(c.dim)
	if err != nil {
		return
	}

	s, err = s.UnSqueeze(c.dim)
	if err != nil {
		return
	}

	return x.Div(s)
}

/* ----- helpers ----- */

func (c *Softmax) toValidInputs(xs []tensor.Tensor) (x tensor.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected exactly one input tensor: got (%d)", len(xs))
		return
	}

	x = xs[0]

	shape := x.Shape()

	if len(shape) <= c.dim {
		err = fmt.Errorf("expected input tensor shape to match 'Dim': %v !~ (%d)", shape, c.dim)
		return
	}

	return x, nil
}

func toValidSoftmaxConfig(iconf *SoftmaxConfig) (conf *SoftmaxConfig, err error) {
	if iconf == nil {
		iconf = &SoftmaxConfig{
			Dim: softmaxDefaultDim,
		}
	}

	conf = new(SoftmaxConfig)
	*conf = *iconf

	if conf.Dim < 0 {
		err = fmt.Errorf("expected 'Dim' not to be negative: got (%d)", conf.Dim)
		return
	}

	return conf, nil
}
