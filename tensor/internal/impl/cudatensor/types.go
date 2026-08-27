//go:build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR}/lib -Wl,-rpath=${SRCDIR}/lib -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import (
	"sync"
	"unsafe"

	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
)

type CUDATensor struct {
	ofst int
	strd []int
	dims []int
	sbuf *sharedBuffer
	gctx *gradtrack.GradContext
}

type sharedBuffer struct {
	data unsafe.Pointer
	size int
	rcnt int
	mutx sync.Mutex
}

type unaryOperatorFunc_C func(C.CUDATensor, C.CUDAView) *C.double
type binaryOperatorFunc_C func(C.CUDATensor, C.CUDATensor, C.CUDAView) *C.double
type halfBinaryOperatorFunc_C func(C.CUDATensor, C.double, C.CUDAView) *C.double
type reducerFunc_C func(C.CUDATensor) C.double
type dimReducerFunc_C func(C.CUDATensor, C.int, C.CUDAView) *C.double
