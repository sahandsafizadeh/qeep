package core

const MaxDims = 6

type Config struct {
	Device    Device
	GradTrack bool
}

type InputDataType interface {
	float64 |
		[]float64 |
		[][]float64 |
		[][][]float64 |
		[][][][]float64
}
