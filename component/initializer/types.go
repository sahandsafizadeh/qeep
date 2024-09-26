package initializers

import qt "github.com/sahandsafizadeh/qeep/tensor"

type Initializer interface {
	Init(shape []int32) (qt.Tensor, error)
}
