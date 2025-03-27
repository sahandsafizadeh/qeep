package cudatensor

import (
	"unsafe"

	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
)

type CUDATensor struct {
	dims []int
	data unsafe.Pointer
	gctx *gradtrack.GradContext
}
