package layers

import "github.com/sahandsafizadeh/qeep/tensor"

type Initializer interface {
	Init(shape []int) (tensor.Tensor, error)
}

type Weight struct {
	Value     *tensor.Tensor
	Trainable bool
}