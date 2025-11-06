package tensor

import "strconv"

type Device int

const (
	CPU Device = iota + 1
	CUDA
)

func (d Device) String() string {
	switch d {
	case CPU:
		return "CPU"
	case CUDA:
		return "CUDA"
	default:
		return strconv.Itoa(int(d))
	}
}
