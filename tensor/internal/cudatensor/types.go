package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR} -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import (
	"unsafe"

	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
)

type CUDATensor struct {
	n    int
	dims []int
	data unsafe.Pointer
	gctx *gradtrack.GradContext
}

type cudacUnaryFunc func(C.CudaData) *C.double
type cudacBinaryFunc func(C.CudaData, C.CudaData) *C.double
type cudacHalfBinaryFunc func(C.CudaData, C.double) *C.double
type cudacReducerFunc func(*C.double, C.size_t) C.double
