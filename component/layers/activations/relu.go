package activations

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Relu struct {
}

func NewRelu() *Relu {
	return &Relu{}
}

func (c *Relu) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	x, err := c.toValidInputs(xs)
	if err != nil {
		return y, fmt.Errorf("Relu input data validation failed: %w", err)
	}

	y, err = c.forward(x)
	if err != nil {
		return y, fmt.Errorf("Relu forward failed: %w", err)
	}

	return y, nil
}

func (c *Relu) forward(x tensor.Tensor) (y tensor.Tensor, err error) {
	_0, err := c.toUntrackedFull(x, 0)
	if err != nil {
		return y, err
	}

	return _0.ElMax(x)
}

/* ----- helpers ----- */

func (c *Relu) toValidInputs(xs []tensor.Tensor) (x tensor.Tensor, err error) {
	if len(xs) != 1 {
		return x, fmt.Errorf("expected exactly one input tensor: got (%d)", len(xs))
	}

	x = xs[0]

	return x, nil
}

func (c *Relu) toUntrackedFull(x tensor.Tensor, value float64) (y tensor.Tensor, err error) {
	dev := x.Device()
	dims := x.Shape()

	return tensor.Full(dims, value, &tensor.Config{
		Device:    dev,
		GradTrack: false,
	})
}
