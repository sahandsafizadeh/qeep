package initialization

import (
	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func Uniform(conf *tinit.Config, l, u float64, shape []int32) (x qt.Tensor, err error) {
	return tinit.RandU(conf, l, u, shape...)
}

func Normal(conf *tinit.Config, u, s float64, shape []int32) (x qt.Tensor, err error) {
	return tinit.RandN(conf, u, s, shape...)
}
