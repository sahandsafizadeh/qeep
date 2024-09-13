package activation

import (
	"fmt"

	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type LeakyRelu struct {
	m float64 // default = 0.01
}

func NewLeakyRelu(m float64) (c *LeakyRelu) {
	return &LeakyRelu{
		m: m,
	}
}

func (c *LeakyRelu) Forward(xs ...qt.Tensor) (y qt.Tensor, err error) {
	x, err := toValidLeakyReluInputs(xs)
	if err != nil {
		return
	}

	return c.forward(x)
}

func (c *LeakyRelu) forward(x qt.Tensor) (y qt.Tensor, err error) {
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

func toValidLeakyReluInputs(xs []qt.Tensor) (x qt.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected leaky relu to receive exactly one input: got (%d)", len(xs))
		return
	}

	return xs[0], nil
}
