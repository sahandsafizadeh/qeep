package activation

import (
	"fmt"

	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type Softmax struct {
	dim int32
}

type SoftmaxConfig struct {
	Dim int32
}

const softmaxDefaultDim = 0

func NewSoftmax(conf *SoftmaxConfig) (c *Softmax, err error) {
	conf, err = toValidSoftmaxConfig(conf)
	if err != nil {
		return
	}

	return &Softmax{
		dim: conf.Dim,
	}, nil
}

func (c *Softmax) Forward(xs ...qt.Tensor) (y qt.Tensor, err error) {
	x, err := toValidSoftmaxInputs(xs)
	if err != nil {
		return
	}

	return c.forward(x)
}

func (c *Softmax) forward(x qt.Tensor) (y qt.Tensor, err error) {
	x = x.Exp()

	s, err := x.SumAlong(c.dim)
	if err != nil {
		return
	}

	return x.Div(s)
}

/* ----- helpers ----- */

func toValidSoftmaxConfig(iconf *SoftmaxConfig) (conf *SoftmaxConfig, err error) {
	if iconf == nil {
		iconf = &SoftmaxConfig{
			Dim: softmaxDefaultDim,
		}
	}

	conf = new(SoftmaxConfig)
	*conf = *iconf

	if conf.Dim < 0 {
		err = fmt.Errorf("expected softmax 'Dim' not to be negative: got (%d)", conf.Dim)
		return
	}

	return conf, nil
}

func toValidSoftmaxInputs(xs []qt.Tensor) (x qt.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected softmax to receive exactly one input: got (%d)", len(xs))
		return
	}

	return xs[0], nil
}
