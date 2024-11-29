package types

import (
	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

type Forwarder interface {
	Forward(...tensor.Tensor) (tensor.Tensor, error)
}

type WeightedForwarder interface {
	Forwarder
	Weights() []layers.Weight
}

type LossFunction interface {
	Compute(yp tensor.Tensor, yt tensor.Tensor) (tensor.Tensor, error)
}

type Optimizer interface {
	Update(*tensor.Tensor) error
}

type Metric interface {
	Accumulate(yp tensor.Tensor, yt tensor.Tensor) error
	Result() (float64, error)
}

type BatchGenerator interface {
	Reset()
	Count() int
	HasNext() bool
	NextBatch() (xs []tensor.Tensor, y tensor.Tensor, err error)
}
