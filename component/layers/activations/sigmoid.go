package activations

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Sigmoid struct {
}

func NewSigmoid() (c *Sigmoid) {
	return &Sigmoid{}
}

func (c *Sigmoid) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	x, err := c.toValidInputs(xs)
	if err != nil {
		err = fmt.Errorf("Sigmoid input data validation failed: %w", err)
		return
	}

	return c.forward(x)
}

func (c *Sigmoid) forward(x tensor.Tensor) (y tensor.Tensor, err error) {
	_1, err := c.toUntrackedFull(x, 1)
	if err != nil {
		return
	}

	x = x.Scale(-1)
	x = x.Exp()

	y, err = _1.Add(x)
	if err != nil {
		return
	}

	return y.Pow(-1), nil
}

/* ----- helpers ----- */

func (c *Sigmoid) toValidInputs(xs []tensor.Tensor) (x tensor.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected exactly one input tensor: got (%d)", len(xs))
		return
	}

	x = xs[0]

	return x, nil
}

func (c *Sigmoid) toUntrackedFull(x tensor.Tensor, value float64) (y tensor.Tensor, err error) {
	dev := x.Device()
	dims := x.Shape()

	return tensor.Full(dims, value, &tensor.Config{
		Device:    dev,
		GradTrack: false,
	})
}
