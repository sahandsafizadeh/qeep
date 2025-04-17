package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR} -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import (
	"unsafe"

	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
)

func (t *CUDATensor) at(index []int) (elem float64) {
	dimsptr := intSliceToCptr(t.dims)
	indexptr := intSliceToCptr(index)
	n := len(index)

	data_c := (*C.double)(t.data)
	dims_c := (*C.int)(dimsptr)
	index_c := (*C.int)(indexptr)
	n_c := (C.size_t)(n)

	elem_c := C.At(data_c, dims_c, index_c, n_c)

	return float64(elem_c)
}

func (t *CUDATensor) slice(index []tensor.Range) (o *CUDATensor) {
	index = completeIndex(index, t.dims)
	dimsptr := intSliceToCptr(t.dims)
	indexptr := rangeSliceToCptr(index)
	n := len(index)

	src_c := (*C.double)(t.data)
	dims_c := (*C.int)(dimsptr)
	index_c := (*C.Range)(indexptr)
	n_c := (C.size_t)(n)

	data_c := C.Slice(src_c, dims_c, index_c, n_c)

	dims := indexToDims(index)

	return newCUDATensor(dims, data_c)
}

func (t *CUDATensor) patch(index []tensor.Range, u *CUDATensor) (o *CUDATensor) {
	index = completeIndex(index, t.dims)
	dimsptr := intSliceToCptr(t.dims)
	indexptr := rangeSliceToCptr(index)
	n := len(index)

	bas_c := (*C.double)(t.data)
	dims_c := (*C.int)(dimsptr)
	src_c := (*C.double)(u.data)
	index_c := (*C.Range)(indexptr)
	n_c := (C.size_t)(n)

	data_c := C.Patch(bas_c, dims_c, src_c, index_c, n_c)

	dims := t.dims

	return newCUDATensor(dims, data_c)
}

func completeIndex(index []tensor.Range, dims []int) (cidx []tensor.Range) {
	cidx = make([]tensor.Range, len(dims))
	for i := range cidx {
		// special case of all elements along dim
		if i >= len(index) || (index[i].From == 0 && index[i].To == 0) {
			cidx[i] = tensor.Range{From: 0, To: dims[i]}
		} else {
			cidx[i] = index[i]
		}
	}

	return cidx
}

/* ----- helpers ----- */

func dimsToNumElems(dims []int) (elems int) {
	elems = 1
	for _, dim := range dims {
		elems *= dim
	}

	return elems
}

func indexToDims(index []tensor.Range) (dims []int) {
	dims = make([]int, len(index))
	for i, idx := range index {
		dims[i] = idx.To - idx.From
	}

	return dims
}

func intSliceToCptr(src []int) (ptr unsafe.Pointer) {
	if len(src) == 0 {
		return nil
	}

	dst := make([]C.int, len(src))
	for i, e := range src {
		dst[i] = (C.int)(e)
	}

	return (unsafe.Pointer)(&dst[0])
}

func rangeSliceToCptr(src []tensor.Range) (ptr unsafe.Pointer) {
	if len(src) == 0 {
		return nil
	}

	dst := make([]C.Range, len(src))
	for i, e := range src {
		dst[i] = C.Range{
			from: (C.int)(e.From),
			to:   (C.int)(e.To),
		}
	}

	return (unsafe.Pointer)(&dst[0])
}
