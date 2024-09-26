package activations

import (
	"fmt"

	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type Tanh struct {
}

func NewTanh() (c *Tanh) {
	return &Tanh{}
}

func (c *Tanh) Forward(xs ...qt.Tensor) (y qt.Tensor, err error) {
	x, err := toValidTanhInputs(xs)
	if err != nil {
		return
	}

	return c.forward(x)
}

func (c *Tanh) forward(x qt.Tensor) (y qt.Tensor, err error) {
	return x.Tanh(), nil
}

/* ----- helpers ----- */

func toValidTanhInputs(xs []qt.Tensor) (x qt.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected tanh to receive exactly one input: got (%d)", len(xs))
		return
	}

	return xs[0], nil
}
