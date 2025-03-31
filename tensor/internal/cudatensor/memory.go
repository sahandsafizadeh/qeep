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
	tdims := make([]int, len(dims))
	tdata := unsafe.Pointer(data)
	copy(tdims, dims)

	t = &CUDATensor{
		dims: tdims,
		data: tdata,
	}

	runtime.GC()
	runtime.AddCleanup(&t, freeCUDATensorData, data)

	return t
}

func freeCUDATensorData(data *C.double) {
	C.FreeCudaMem(data)
}
