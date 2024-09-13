package component

import (
	"fmt"

	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type Sigmoid struct {
}

func NewSigmoid() (c *Sigmoid) {
	return &Sigmoid{}
}

func (c *Sigmoid) Forward(xs ...qt.Tensor) (y qt.Tensor, err error) {
	x, err := toValidSigmoidInputs(xs)
	if err != nil {
		return
	}

	return c.forward(x)
}

func (c *Sigmoid) forward(x qt.Tensor) (y qt.Tensor, err error) {
	_1 := x.Pow(0)
	x = x.Scale(-1)
	x = x.Exp()

	y, err = _1.Add(x)
	if err != nil {
		return
	}

	return y.Pow(-1), nil
}

/* ----- helpers ----- */

func toValidSigmoidInputs(xs []qt.Tensor) (x qt.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected sigmoid to receive exactly one input: got (%d)", len(xs))
		return
	}

	return xs[0], nil
}
