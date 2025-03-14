package activations

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type LeakyRelu struct {
	m float64
}

type LeakyReluConfig struct {
	M float64
}

const LeakyReluDefaultM = 0.01

func NewLeakyRelu(conf *LeakyReluConfig) (c *LeakyRelu) {
	conf = toValidLeakyReluConfig(conf)

	return &LeakyRelu{
		m: conf.M,
	}
}

func (c *LeakyRelu) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	x, err := c.toValidInputs(xs)
	if err != nil {
		err = fmt.Errorf("LeakyRelu input data validation failed: %w", err)
		return
	}

	return c.forward(x)
}

func (c *LeakyRelu) forward(x tensor.Tensor) (y tensor.Tensor, err error) {
	_0 := x.Scale(0)

	s1, err := _0.ElMax(x)
	if err != nil {
		return
	}

	s2, err := _0.ElMin(x)
	if err != nil {
		return
	}

	s2 = s2.Scale(c.m)

	return s1.Add(s2)
}

/* ----- helpers ----- */

func (c *LeakyRelu) toValidInputs(xs []tensor.Tensor) (x tensor.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected exactly one input tensor: got (%d)", len(xs))
		return
	}

	x = xs[0]

	return x, nil
}

func toValidLeakyReluConfig(iconf *LeakyReluConfig) (conf *LeakyReluConfig) {
	if iconf == nil {
		iconf = &LeakyReluConfig{
			M: LeakyReluDefaultM,
		}
	}

	conf = new(LeakyReluConfig)
	*conf = *iconf

	return conf
}
