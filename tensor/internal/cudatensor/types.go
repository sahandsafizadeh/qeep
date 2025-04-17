package cudatensor

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

type cudacUnaryFunc func(*C.double, C.size_t) *C.double
type cudacBinaryFunc func(*C.double, *C.double, C.size_t) *C.double
type cudacHalfBinaryFunc func(*C.double, C.size_t, C.double) *C.double
