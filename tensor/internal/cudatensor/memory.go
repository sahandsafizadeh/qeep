package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR} -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import (
	"runtime"
	"unsafe"
)

func newCUDATensor(dims []int, data *C.double) (t *CUDATensor) {
	tn := dimsToNumElems(dims)
	tdims := make([]int, len(dims))
	tdata := unsafe.Pointer(data)
	copy(tdims, dims)

	t = &CUDATensor{
		n:    tn,
		dims: tdims,
		data: tdata,
	}

	runtime.AddCleanup(&t, freeCUDATensorData, data)
	if enforceCleanup() {
		runtime.GC()
	}

	return t
}

func freeCUDATensorData(data *C.double) {
	C.FreeCudaMem(data)
}

func enforceCleanup() bool {
	return true
}
