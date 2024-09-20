package initialization

import (
	"math"

	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func HeUniform(conf *tinit.Config, fin int32, shape []int32) (x qt.Tensor, err error) {
	r := math.Sqrt(6. / float64(fin))
	return tinit.RandU(conf, -r, r, shape...)
}

func HeNormal(conf *tinit.Config, fin int32, shape []int32) (x qt.Tensor, err error) {
	s := math.Sqrt(2. / float64(fin))
	return tinit.RandN(conf, 0., s, shape...)
}
