package activations

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Relu struct {
}

func NewRelu() (c *Relu) {
	return &Relu{}
}

func (c *Relu) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	x, err := c.toValidInputs(xs)
	if err != nil {
		err = fmt.Errorf("Relu input data validation failed: %w", err)
		return
	}

	return c.forward(x)
}

func (c *Relu) forward(x tensor.Tensor) (y tensor.Tensor, err error) {
	_0 := x.Scale(0)

	return _0.ElMax(x)
}

/* ----- helpers ----- */

func (c *Relu) toValidInputs(xs []tensor.Tensor) (x tensor.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected exactly one input tensor: got (%d)", len(xs))
		return
	}

	x = xs[0]

	return x, nil
}
