package tensor

import "github.com/sahandsafizadeh/qeep/tensor/internal/tensor"

type Tensor = tensor.Tensor
type Device = tensor.Device
type Range = tensor.Range

const (
	CPU  = tensor.CPU
	CUDA = tensor.CUDA
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
