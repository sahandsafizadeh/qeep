package initializers

import (
	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

type Initializer interface {
	Init(dev tinit.Device, shape []int32) (qt.Tensor, error)
}
