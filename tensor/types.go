package tensor

import "github.com/sahandsafizadeh/qeep/tensor/internal/tensor"

type Tensor = tensor.Tensor
type Range = tensor.Range

type Device int

const (
	CPU Device = iota + 1
	CUDA
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
