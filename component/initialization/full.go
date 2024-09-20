package initialization

import (
	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func Full(conf *tinit.Config, value float64, shape []int32) (x qt.Tensor, err error) {
	return tinit.Full(conf, value, shape...)
}
