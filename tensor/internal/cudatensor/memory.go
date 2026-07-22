//go:build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR}/lib -Wl,-rpath=${SRCDIR}/lib -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import (
	"math/rand/v2"
	"runtime"
	"sync"
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
	cudaMemMutx  sync.Mutex
)

func newCUDATensor(dims []int, data *C.double) *CUDATensor {
	t := new(CUDATensor)
	t.ofst = 0
	t.strd = util.DimsToStrides(dims)
	t.dims = make([]int, len(dims))
	copy(t.dims, dims)

	sbuf := new(sharedBuffer)
	sbuf.data = unsafe.Pointer(data)
	sbuf.size = util.DimsToNumElems(dims)
	sbuf.rcnt = 1

	t.sbuf = sbuf
	runtime.AddCleanup(t, freeCUDATensorData, sbuf)

	updateCudaAllocMem(sbuf.size, +1)
	if enforceCleanup() {
		runtime.GC()
	}

	return t
}

func shareCUDATensorData(dst *CUDATensor, src *CUDATensor) {
	src.sbuf.mutx.Lock()
	src.sbuf.rcnt++
	src.sbuf.mutx.Unlock()

	sbuf := src.sbuf
	// keep src reachable until after increment

	dst.sbuf = sbuf
	runtime.AddCleanup(dst, freeCUDATensorData, sbuf)
}

func freeCUDATensorData(sbuf *sharedBuffer) {
	sbuf.mutx.Lock()
	sbuf.rcnt--
	release := sbuf.rcnt == 0
	sbuf.mutx.Unlock()

	if release {
		C.FreeCudaMem((*C.double)(sbuf.data))
		updateCudaAllocMem(sbuf.size, -1)
	}
}

func updateCudaAllocMem(n int, dir int) {
	cudaMemMutx.Lock()
	defer cudaMemMutx.Unlock()

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
	cudaMemMutx.Lock()
	defer cudaMemMutx.Unlock()

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

func memInfoInitialized() bool {
	return cudaTotalMem > 0
}

func shouldReloadByChance() bool {
	return rand.Float64() < memInfoReloadRate
}
