package component

import (
	"fmt"

	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type Component interface {
	Forward(...qt.Tensor) (qt.Tensor, error)
}

type WeightedComponent interface {
	Component
	Weights() []*qt.Tensor
}

type LossFunc func(yp qt.Tensor, yt qt.Tensor) (qt.Tensor, error)
type OptimizerFunc func(*qt.Tensor) error

type FC struct {
	w qt.Tensor
	b qt.Tensor
}

func (c *FC) Forward(x qt.Tensor) (y qt.Tensor, err error) {
	err = validateFCX(x)
	if err != nil {
		return
	}

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

func validateFCX(x qt.Tensor) (err error) {
	nd := len(x.Shape())
	if nd != 2 {
		err = fmt.Errorf("expected FC layer's x input to have (2) dimensions for batch*xd: got (%d)", nd)
		return
	}

	return nil
}

// dense, conv1d, conv2d, conv3d, lstm, gru, embedding, batchnorm, layernorm, multiheadattention,
// conv2dtranspose, attention
