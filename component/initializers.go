package component

import (
	"math"

	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func FullInit(conf *tinit.Config, value float64, shape []int32) (x qt.Tensor, err error) {
	return tinit.Full(conf, value, shape...)
}

func UniformInit(conf *tinit.Config, l, u float64, shape []int32) (x qt.Tensor, err error) {
	return tinit.RandU(conf, l, u, shape...)
}

func NormalInit(conf *tinit.Config, u, s float64, shape []int32) (x qt.Tensor, err error) {
	return tinit.RandN(conf, u, s, shape...)
}

func HeUniformInit(conf *tinit.Config, fin int32, shape []int32) (x qt.Tensor, err error) {
	r := math.Sqrt(6. / float64(fin))
	return tinit.RandU(conf, -r, r, shape...)
}

func HeNormalInit(conf *tinit.Config, fin int32, shape []int32) (x qt.Tensor, err error) {
	s := math.Sqrt(2. / float64(fin))
	return tinit.RandN(conf, 0., s, shape...)
}

func XavierUniformInit(conf *tinit.Config, fin, fout int32, shape []int32) (x qt.Tensor, err error) {
	r := math.Sqrt(6. / float64(fin+fout))
	return tinit.RandU(conf, -r, r, shape...)
}

func XavierNormalInit(conf *tinit.Config, fin, fout int32, shape []int32) (x qt.Tensor, err error) {
	s := math.Sqrt(2. / float64(fin+fout))
	return tinit.RandN(conf, 0., s, shape...)
}
