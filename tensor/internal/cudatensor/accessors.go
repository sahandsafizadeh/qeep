//go:build cuda
// +build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR}/lib -Wl,-rpath=${SRCDIR}/lib -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import (
	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

func (t *CUDATensor) at(index []int) (elem float64) {
	data_c := getCudaDataOf(t)
	dims_c := getDimArrOf(t.dims)
	index_c := getDimArrOf(index)

	elem_c := C.At(data_c, dims_c, index_c)

	return float64(elem_c)
}

func (t *CUDATensor) slice(index []tensor.Range) (o *CUDATensor) {
	index = util.CompleteIndex(index, t.dims)
	dims := util.IndexToDims(index)

	src_c := getCudaDataOf(t)
	dims_c := getDimArrOf(t.dims)
	index_c := getRangeArrOf(index)

	data_c := C.Slice(src_c, dims_c, index_c)

	return newCUDATensor(dims, data_c)
}

func (t *CUDATensor) patch(index []tensor.Range, u *CUDATensor) (o *CUDATensor) {
	index = util.CompleteIndex(index, u.dims)
	dims := t.dims

	bas_c := getCudaDataOf(t)
	dims_c := getDimArrOf(t.dims)
	src_c := getCudaDataOf(u)
	index_c := getRangeArrOf(index)

	data_c := C.Patch(bas_c, dims_c, src_c, index_c)

	return newCUDATensor(dims, data_c)
}
