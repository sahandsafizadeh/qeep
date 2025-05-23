//go:build cuda
// +build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR}/lib -Wl,-rpath=${SRCDIR}/lib -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import (
	"math/rand/v2"
	"runtime"
	"unsafe"

	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

const (
	allocatedMemThreshold = 0.75
	memInfoReloadRate     = 1e-4
	doubleSizeBytes       = 8
)

var (
	cudaTotalMem int64 = 0
	cudaAllocMem int64 = 0
)

func newCUDATensor(dims []int, data *C.double) (t *CUDATensor) {
	tn := util.DimsToNumElems(dims)
	tdims := make([]int, len(dims))
	tdata := unsafe.Pointer(data)
	copy(tdims, dims)

	t = &CUDATensor{
		n:    tn,
		dims: tdims,
		data: tdata,
	}

	arg := getCudaDataOf(t)
	updateCudaAllocMem(t.n, +1)
	runtime.AddCleanup(t, freeCUDATensorData, arg)

	if enforceCleanup() {
		runtime.GC()
	}

	return t
}

func freeCUDATensorData(cd C.CudaData) {
	C.FreeCudaMem(cd.arr)
	updateCudaAllocMem(int(cd.size), -1)
}

func updateCudaAllocMem(n int, dir int) {
	updatedBytes := int64(n) * doubleSizeBytes
	if dir > 0 {
		cudaAllocMem += updatedBytes
	} else {
		cudaAllocMem -= updatedBytes
	}

	if cudaAllocMem < 0 {
		panic("reached negative value for cudaAllocMem")
	}
}

func enforceCleanup() bool {
	reloadCudaMemInfo()
	allocMem := float64(cudaAllocMem)
	totalMem := float64(cudaTotalMem)
	return allocMem/totalMem > allocatedMemThreshold
}

func reloadCudaMemInfo() {
	if !memInfoInitialized() || shouldReloadByChance() {
		var freemem C.size_t
		var totalmem C.size_t

		freeptr := unsafe.Pointer(&freemem)
		totalptr := unsafe.Pointer(&totalmem)

		free_mem_c := (*C.size_t)(freeptr)
		total_mem_c := (*C.size_t)(totalptr)

		C.GetCudaMemInfo(free_mem_c, total_mem_c)

		cudaFreeMem := int64(freemem)
		cudaTotalMem = int64(totalmem)
		cudaAllocMem = cudaTotalMem - cudaFreeMem
	}
}

/* ----- helpers ----- */

func memInfoInitialized() bool {
	return cudaTotalMem > 0
}

func shouldReloadByChance() bool {
	return rand.Float64() < memInfoReloadRate
}
