package activations

import (
	"fmt"

	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type Relu struct {
}

func NewRelu() (c *Relu) {
	return &Relu{}
}

func (c *Relu) Forward(xs ...qt.Tensor) (y qt.Tensor, err error) {
	x, err := toValidReluInputs(xs)
	if err != nil {
		return
	}

	return c.forward(x)
}

func (c *Relu) forward(x qt.Tensor) (y qt.Tensor, err error) {
	_0 := x.Scale(0)

	return _0.ElMax(x)
}

/* ----- helpers ----- */

func toValidReluInputs(xs []qt.Tensor) (x qt.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected relu to receive exactly one input: got (%d)", len(xs))
		return
	}

	return xs[0], nil
}
