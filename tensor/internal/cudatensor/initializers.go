package cudatensor

/*
   #cgo LDFLAGS: -L. -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import (
	"runtime"
	"unsafe"
)

func newCUDATensor(dims []int, data *C.double) (t *CUDATensor) {
	tdims := make([]int, len(dims))
	tdata := unsafe.Pointer(data)
	copy(tdims, dims)

	t = &CUDATensor{
		dims: tdims,
		data: tdata,
	}

	runtime.AddCleanup(&t, freeCUDATensorData, data)

	return t
}

func freeCUDATensorData(data *C.double) {
	C.FreeCUDAMemory(data)
}

func constTensor(dims []int, val float64) (t *CUDATensor) {
	n := (C.size_t)(dimsToNumElems(dims))
	value := (C.double)(val)

	data := C.Full(n, value)

	return newCUDATensor(dims, data)
}
