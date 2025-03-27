package cudatensor

/*
   #cgo LDFLAGS: -L. -linitializers
   #include "initializers.h"
*/
import "C"

func constTensor(value float64, dims []int) (t *CUDATensor) {
	n := (C.size_t)((&CUDATensor{dims: dims}).numElems())
	v := (C.double)(value)

	data := C.Full(n, v)

	return newCUDATensor(dims, data)
}
