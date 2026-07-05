package layers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Add struct {
}

func NewAdd() *Add {
	return &Add{}
}

func (c *Add) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	x1, x2, err := c.toValidInputs(xs)
	if err != nil {
		return y, fmt.Errorf("Add input data validation failed: %w", err)
	}

	y, err = c.forward(x1, x2)
	if err != nil {
		return y, fmt.Errorf("Add forward failed: %w", err)
	}

	return y, nil
}

func (c *Add) forward(x1, x2 tensor.Tensor) (y tensor.Tensor, err error) {
	return x1.Add(x2)
}

/* ----- helpers ----- */

func (c *Add) toValidInputs(xs []tensor.Tensor) (x1, x2 tensor.Tensor, err error) {
	if len(xs) != 2 {
		return x1, x2, fmt.Errorf("expected exactly two input tensors: got (%d)", len(xs))
	}

	x1 = xs[0]
	x2 = xs[1]

	return x1, x2, nil
}
