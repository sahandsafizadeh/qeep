package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR} -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

func (t *CUDATensor) numElems() (n int) {
	return dimsToNumElems(t.dims)
}

func (t *CUDATensor) at(index []int) (elem float64) {
	position := 0
	dimTotal := 1
	for i := len(index) - 1; i >= 0; i-- {
		position += dimTotal * index[i]
		dimTotal *= t.dims[i]
	}

	data_c := (*C.double)(t.data)
	index_c := (C.size_t)(position)
	elem_c := C.At(data_c, index_c)

	return float64(elem_c)
}

/* ----- helpers ----- */

func dimsToNumElems(dims []int) (elems int) {
	elems = 1
	for _, dim := range dims {
		elems *= dim
	}

	return elems
}
