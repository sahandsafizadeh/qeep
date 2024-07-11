package component

import (
	t "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

type Component interface {
	Allocate() error
	Forward(t.Tensor) (t.Tensor, error)
	UpdateWeights() error
}

type FC struct {
	w t.Tensor
	b t.Tensor
}

func (c *FC) Allocate() (err error) {
	c.w, err = tinit.Zeros(nil)
	if err != nil {
		return
	}

	c.b, err = tinit.Zeros(nil)
	if err != nil {
		return
	}

	return nil
}

func (c *FC) Forward(x t.Tensor) (y t.Tensor, err error) {
	w, err := c.w.UnSqueeze(1)
	if err != nil {
		return
	}

	// last dim - 1
	x, err = x.UnSqueeze(1)
	if err != nil {
		return
	}

	y, err = w.MatMul(x)
	if err != nil {
		return
	}

	// last dim
	y, err = y.SumAlong(2)
	if err != nil {
		return
	}

	y, err = y.Add(c.b)
	if err != nil {
		return
	}

	return y, nil
}

func (c *FC) UpdateWeights() (err error) {
	return
}
