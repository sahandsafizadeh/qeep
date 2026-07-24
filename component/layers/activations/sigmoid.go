package activations

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Sigmoid struct {
}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (c *Sigmoid) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	x, err := c.toValidInputs(xs)
	if err != nil {
		return y, fmt.Errorf("Sigmoid input data validation failed: %w", err)
	}

	y, err = c.forward(x)
	if err != nil {
		return y, fmt.Errorf("Sigmoid forward failed: %w", err)
	}

	return y, nil
}

func (c *Sigmoid) forward(x tensor.Tensor) (y tensor.Tensor, err error) {
	/*
		For numerical stability:
		sigmoid(x) =
			- x >= 0: 1 / (1 + exp(-x))
			- x <  0: exp(x) / (1 + exp(x))
	*/

	_0, err := c.toUntrackedFull(x, 0)
	if err != nil {
		return y, err
	}
	_1, err := c.toUntrackedFull(x, 1)
	if err != nil {
		return y, err
	}

	xp, err := x.ElMax(_0)
	if err != nil {
		return y, err
	}
	xn, err := x.ElMin(_0)
	if err != nil {
		return y, err
	}

	maskp, err := x.Ge(_0)
	if err != nil {
		return y, err
	}
	maskn, err := _1.Sub(maskp)
	if err != nil {
		return y, err
	}

	// ----- positive part -----

	posr := xp.Scale(-1).Exp()
	posr, err = _1.Add(posr)
	if err != nil {
		return y, err
	}
	posr = posr.Pow(-1)

	posr, err = posr.Mul(maskp)
	if err != nil {
		return y, err
	}

	// ----- negative part -----

	nn := xn.Exp()
	nd, err := _1.Add(nn)
	if err != nil {
		return y, err
	}
	negr, err := nn.Div(nd)
	if err != nil {
		return y, err
	}

	negr, err = negr.Mul(maskn)
	if err != nil {
		return y, err
	}

	// ----- final result -----

	return posr.Add(negr)
}

/* ----- helpers ----- */

func (c *Sigmoid) toValidInputs(xs []tensor.Tensor) (x tensor.Tensor, err error) {
	if len(xs) != 1 {
		return x, fmt.Errorf("expected exactly one input tensor: got (%d)", len(xs))
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
