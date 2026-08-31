//go:build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR}/lib -Wl,-rpath=${SRCDIR}/lib -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import (
	"unsafe"

	"github.com/sahandsafizadeh/qeep/tensor/internal/core"
	"github.com/sahandsafizadeh/qeep/tensor/internal/impl/common/dimsutil"
)

func (t *CUDATensor) numElems() int {
	return dimsutil.DimsToNumElems(t.dims)
}

func (t *CUDATensor) at(index []int) float64 {
	t_c := toCUDATensor_C(t)
	index_c := toDimArr_C(index)

	elem_c := C.At(t_c, index_c)

	return float64(elem_c)
}

func (t *CUDATensor) slice(index []core.Range) *CUDATensor {
	cidx := dimsutil.CompleteIndex(index, t.dims)

	o := new(CUDATensor)
	o.ofst = t.ofst
	o.strd = make([]int, len(t.strd))
	copy(o.strd, t.strd)
	o.dims = dimsutil.IndexToDims(cidx)

	for i, r := range cidx {
		o.ofst += t.strd[i] * r.From
	}

	shareCUDATensorData(o, t) // reuse data

	return o
}

func (t *CUDATensor) export() *core.Snapshot {
	n := t.numElems()
	output_data := make([]C.double, n)

	output_data_ptr := unsafe.Pointer(&output_data[0])

	t_c := toCUDATensor_C(t)
	output_data_c := (*C.double)(output_data_ptr)
	view_o_c := toCUDAView_C(t.dims)

	C.Export(t_c, output_data_c, view_o_c)

	dims := make([]int, len(t.dims))
	copy(dims, t.dims)

	data := make([]float64, len(output_data))
	for i, v := range output_data {
		data[i] = float64(v)
	}

	return &core.Snapshot{
		Dims: dims,
		Data: data,
	}
}
