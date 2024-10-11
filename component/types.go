package component

import qt "github.com/sahandsafizadeh/qeep/tensor"

type Initializer interface {
	Init(shape []int) (qt.Tensor, error)
}

type Forwarder interface {
	Forward(...qt.Tensor) (qt.Tensor, error)
}

type WeightedForwarder interface {
	Forwarder
	TrainableWeights() []*qt.Tensor
}

type LossFunction interface {
	Compute(yp qt.Tensor, yt qt.Tensor) (qt.Tensor, error)
}

type Optimizer interface {
	Update(*qt.Tensor) error
}
