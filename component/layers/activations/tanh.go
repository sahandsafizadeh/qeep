package activations

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Tanh struct {
}

func NewTanh() (c *Tanh) {
	return &Tanh{}
}

func (c *Tanh) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	x, err := c.toValidInputs(xs)
	if err != nil {
		err = fmt.Errorf("Tanh input data validation failed: %w", err)
		return
	}

	return c.forward(x)
}

func (c *Tanh) forward(x tensor.Tensor) (y tensor.Tensor, err error) {
	return x.Tanh(), nil
}

/* ----- helpers ----- */

func (c *Tanh) toValidInputs(xs []tensor.Tensor) (x tensor.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected exactly one input tensor: got (%d)", len(xs))
		return
	}

	x = xs[0]

	return x, nil
}
