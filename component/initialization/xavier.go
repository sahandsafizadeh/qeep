package initialization

import (
	"math"

	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func XavierUniform(conf *tinit.Config, fin, fout int32, shape []int32) (x qt.Tensor, err error) {
	r := math.Sqrt(6. / float64(fin+fout))
	return tinit.RandU(conf, -r, r, shape...)
}

func XavierNormal(conf *tinit.Config, fin, fout int32, shape []int32) (x qt.Tensor, err error) {
	s := math.Sqrt(2. / float64(fin+fout))
	return tinit.RandN(conf, 0., s, shape...)
}
