//go:build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR}/lib -Wl,-rpath=${SRCDIR}/lib -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import (
	"unsafe"

	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
)

type CUDATensor struct {
	ofst int
	strd []int
	dims []int
	data unsafe.Pointer
	gctx *gradtrack.GradContext
}

type unaryOperatorFunc_C func(C.CUDATensor, C.CUDAView) *C.double
type binaryOperatorFunc_C func(C.CUDATensor, C.CUDATensor, C.CUDAView) *C.double
type halfBinaryOperatorFunc_C func(C.CUDATensor, C.double, C.CUDAView) *C.double
type reducerFunc_C func(C.CUDATensor) C.double
type dimReducerFunc_C func(C.CUDATensor, C.int, C.CUDAView) *C.double
