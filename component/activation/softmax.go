package activation

import (
	"fmt"

	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type Softmax struct {
	dim int32 // default = 0
}

func NewSoftmax(dim int32) (c *Softmax) {
	return &Softmax{
		dim: dim,
	}
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

func toValidSoftmaxInputs(xs []qt.Tensor) (x qt.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected softmax to receive exactly one input: got (%d)", len(xs))
		return
	}

	return xs[0], nil
}
