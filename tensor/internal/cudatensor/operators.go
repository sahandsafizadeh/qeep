package cudatensor

/*
   #cgo LDFLAGS: -L. -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

func (t *CUDATensor) scale(u float64) (o *CUDATensor) {
	x := (*C.double)(t.data)
	a := (C.double)(u)
	n := (C.size_t)(t.numElems())

	data := C.Scale(x, a, n)

	return newCUDATensor(t.dims, data)
}

func (t *CUDATensor) pow(u float64) (o *CUDATensor) {
	x := (*C.double)(t.data)
	a := (C.double)(u)
	n := (C.size_t)(t.numElems())

	data := C.Pow(x, a, n)

	return newCUDATensor(t.dims, data)
}

func (t *CUDATensor) exp() (o *CUDATensor) {
	x := (*C.double)(t.data)
	n := (C.size_t)(t.numElems())

	data := C.Exp(x, n)

	return newCUDATensor(t.dims, data)
}
