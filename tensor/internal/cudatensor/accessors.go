package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR} -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import "unsafe"

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

// todo: wire up slice and patch

/* ----- helpers ----- */

func dimsToNumElems(dims []int) (elems int) {
	elems = 1
	for _, dim := range dims {
		elems *= dim
	}

	return elems
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
