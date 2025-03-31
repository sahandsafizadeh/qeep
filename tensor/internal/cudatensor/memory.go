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

	if callGC() {
		runtime.GC()
	}

	return t
}

func freeCUDATensorData(data *C.double) {
	C.FreeCudaMem(data)
}

var isFirstTime = true
var state float64

func callGC() bool {
	var freeMem C.size_t
	var totalMem C.size_t

	C.GetCudaMemInfo(&totalMem, &freeMem)

	free := float64(freeMem)
	total := float64(totalMem)

	allocated := total - free

	if isFirstTime {
		isFirstTime = false
		state = allocated
		return true
	}

	if allocated/total >= 0.9 {
		// fmt.Println("close to death")
		state = allocated
		return true
	}

	if allocated/state >= 2. {
		// fmt.Println("twice allocated")
		state = allocated
		return true
	}

	if allocated < state {
		state = allocated
	}

	return false
}
