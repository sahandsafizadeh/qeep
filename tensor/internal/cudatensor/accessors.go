//go:build cuda

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

func (t *CUDATensor) numElems() int {
	return util.DimsToNumElems(t.dims)
}

func (t *CUDATensor) at(index []int) float64 {
	t_c := toCUDATensor_C(t)
	index_c := toDimArr_C(index)

	elem_c := C.At(t_c, index_c)

	return float64(elem_c)
}

func (t *CUDATensor) slice(index []tensor.Range) *CUDATensor {
	cidx := util.CompleteIndex(index, t.dims)

	o := new(CUDATensor)
	o.ofst = t.ofst
	o.strd = make([]int, len(t.strd))
	copy(o.strd, t.strd)
	o.dims = util.IndexToDims(cidx)

	for i, r := range cidx {
		o.ofst += t.strd[i] * r.From
	}

	shareCUDATensorData(o, t) // reuse data

	return o
}

func (t *CUDATensor) patch(index []tensor.Range, u *CUDATensor) *CUDATensor {
	index = util.CompleteIndex(index, u.dims)
	dims := t.dims

	t_c := toCUDATensor_C(t)
	ranges_c := toRangeArr_C(index)
	u_c := toCUDATensor_C(u)
	view_o_c := toCUDAView_C(dims)

	data_c := C.Patch(t_c, ranges_c, u_c, view_o_c)

	return newCUDATensor(dims, data_c)
}
