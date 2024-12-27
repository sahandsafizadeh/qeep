package contract

import (
	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

type Layer interface {
	Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error)
}

type WeightedLayer interface {
	Layer
	Weights() []layers.Weight
}

type Loss interface {
	Compute(yp tensor.Tensor, yt tensor.Tensor) (l tensor.Tensor, err error)
}

type Metric interface {
	Accumulate(yp tensor.Tensor, yt tensor.Tensor) (err error)
	Result() float64
}

type Optimizer interface {
	Update(weight *tensor.Tensor) (err error)
}

type BatchGenerator interface {
	Reset()
	Count() int
	HasNext() bool
	NextBatch() (xs []tensor.Tensor, y tensor.Tensor, err error)
}
