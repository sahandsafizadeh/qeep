package tensor

import "github.com/sahandsafizadeh/qeep/tensor/internal/tensor"

// Tensor is a multi-dimensional array supporting linear algebra and automatic differentiation.
type Tensor = tensor.Tensor

// Device selects where tensor data and computation live (CPU or CUDA).
type Device = tensor.Device

// Range specifies a half-open interval [Start, End) for slicing.
type Range = tensor.Range

// Supported devices for tensor allocation and ops.
const (
	CPU  = tensor.CPU
	CUDA = tensor.CUDA
)

// Config specifies options for tensor creation.
// Device selects CPU or CUDA; GradTrack enables automatic differentiation.
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
