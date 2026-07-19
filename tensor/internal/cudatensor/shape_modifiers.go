//go:build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR}/lib -Wl,-rpath=${SRCDIR}/lib -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import (
	"slices"

	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

func (t *CUDATensor) transpose() *CUDATensor {
	n := len(t.dims)

	o := new(CUDATensor)
	o.ofst = t.ofst
	o.strd = make([]int, n)
	copy(o.strd, t.strd)
	o.strd[n-2], o.strd[n-1] = o.strd[n-1], o.strd[n-2]
	o.dims = util.TransposeDims(t.dims)

	shareCUDAData(o, t) // reuse data

	return o
}

func (t *CUDATensor) broadcast(shape []int) *CUDATensor {
	o := new(CUDATensor)
	o.ofst = t.ofst
	o.strd = make([]int, len(shape))
	o.dims = make([]int, len(shape))
	copy(o.dims, shape)

	offset := len(shape) - len(t.dims)
	for i := range o.strd {
		if i < offset {
			o.strd[i] = 0
		} else if j := i - offset; t.dims[j] == 1 && shape[i] != 1 {
			o.strd[i] = 0
		} else {
			o.strd[i] = t.strd[j]
		}
	}

	shareCUDAData(o, t) // reuse data

	return o
}

func (t *CUDATensor) reshape(shape []int) *CUDATensor {
	fofst := 0
	fstrd := util.DimsToStrides(t.dims)

	if t.ofst != fofst || !slices.Equal(t.strd, fstrd) { // impossible to reuse data; copy
		t_c := toCUDATensor_C(t)
		view_o_c := toCUDAView_C(shape)

		data_c := C.From(t_c, view_o_c)

		return newCUDATensor(shape, data_c)
	}

	o := new(CUDATensor)
	o.ofst = 0
	o.strd = util.DimsToStrides(shape)
	o.dims = make([]int, len(shape))
	copy(o.dims, shape)

	shareCUDAData(o, t) // reuse data

	return o
}

func (t *CUDATensor) unsqueeze(dim int) *CUDATensor {
	return t.reshape(util.UnSqueezeDims(dim, t.dims))
}

func (t *CUDATensor) squeeze(dim int) *CUDATensor {
	return t.reshape(util.SqueezeDims(dim, t.dims))
}

func (t *CUDATensor) flatten(fromDim int) *CUDATensor {
	return t.reshape(util.FlattenDims(fromDim, t.dims))
}
