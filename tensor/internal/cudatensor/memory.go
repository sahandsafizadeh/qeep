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

type cudaBufferInfo struct {
	size int
	arr  *C.double
}

var cudaBuffers = map[*int]*cudaBufferInfo{}

func newCUDATensor(dims []int, data *C.double) *CUDATensor {
	trefc := 1
	tofst := 0
	tstrd := util.DimsToStrides(dims)
	tdims := make([]int, len(dims))
	copy(tdims, dims)
	tdata := unsafe.Pointer(data)

	t := &CUDATensor{
		refc: &trefc,
		ofst: tofst,
		strd: tstrd,
		dims: tdims,
		data: tdata,
	}

	size := t.numElems()
	cudaBuffers[t.refc] = &cudaBufferInfo{size: size, arr: data}
	updateCudaAllocMem(size, +1)
	runtime.AddCleanup(t, freeCUDATensorData, t.refc)

	if enforceCleanup() {
		runtime.GC()
	}

	return t
}

func shareCUDAData(dst *CUDATensor, src *CUDATensor) {
	dst.refc = src.refc // refc must be a shared resource
	*dst.refc++
	dst.data = src.data

	runtime.AddCleanup(dst, freeCUDATensorData, dst.refc)
}

func freeCUDATensorData(refc *int) {
	*refc--
	if *refc == 0 {
		info := cudaBuffers[refc]
		delete(cudaBuffers, refc)
		C.FreeCudaMem(info.arr)
		updateCudaAllocMem(info.size, -1)
	}
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
