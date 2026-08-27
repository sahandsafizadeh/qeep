package tensor

import "github.com/sahandsafizadeh/qeep/tensor/internal/core"

// Tensor is a multi-dimensional array supporting linear algebra and automatic differentiation.
type Tensor = core.Tensor

// Device selects where tensor data and computation live.
type Device = core.Device

// Range specifies a half-open interval [Start, End) for slicing.
type Range = core.Range

const (
	CPU  = core.CPU
	CUDA = core.CUDA
)

type Config struct {
	Device    Device
	GradTrack bool
}

type inputDataType interface {
	float64 |
		[]float64 |
		[][]float64 |
		[][][]float64 |
		[][][][]float64
}
