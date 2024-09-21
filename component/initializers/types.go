package initializers

import (
	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

type Initializer interface {
	Init(conf *tinit.Config, shape []int32) (qt.Tensor, error)
}
