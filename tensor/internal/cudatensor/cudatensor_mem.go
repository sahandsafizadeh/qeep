package cudatensor

/*
   #cgo LDFLAGS: -L. -lcudatensor
   #include "cudatensor.h"
*/
import "C"

import (
	"runtime"
	"unsafe"
)

func newCUDATensor(dims []int, data *C.double) (t *CUDATensor) {
	tdims := make([]int, len(t.dims))
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
	C.FreeCUDAMem(data)
}
