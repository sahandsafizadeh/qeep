package component

import qt "github.com/sahandsafizadeh/qeep/tensor"

type Component interface {
	Forward(...qt.Tensor) (qt.Tensor, error)
}

type WeightedComponent interface {
	Component
	Weights() []*qt.Tensor
}

type LossFunc func(yp qt.Tensor, yt qt.Tensor) (qt.Tensor, error)
type OptimizerFunc func(*qt.Tensor) error
