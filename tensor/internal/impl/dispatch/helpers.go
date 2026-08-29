package dispatch

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/core"
	"github.com/sahandsafizadeh/qeep/tensor/internal/impl/cputensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/impl/cudatensor"
)

func toValidConfig(iconf *core.Config) (conf *core.Config, err error) {
	if iconf == nil {
		iconf = &core.Config{
			Device:    core.CPU,
			GradTrack: false,
		}
	}

	conf = new(core.Config)
	*conf = *iconf

	switch conf.Device {
	case core.CPU:
	case core.CUDA:
	default:
		return conf, fmt.Errorf("invalid input device")
	}

	return conf, nil
}

func validateImplementation(t core.Tensor) (err error) {
	switch t.(type) {
	case *cputensor.CPUTensor,
		*cudatensor.CUDATensor:
		return nil

	default:
		return fmt.Errorf("unsupported tensor implementation")
	}
}

func validateImplementationsUnity(ts []core.Tensor) (err error) {
	if len(ts) < 2 {
		return fmt.Errorf("expected at least (2) tensors: got (%d)", len(ts))
	}

	var dev core.Device

	for _, t := range ts {
		switch t.(type) {
		case *cputensor.CPUTensor:
			if dev == 0 {
				dev = core.CPU
			} else if dev != core.CPU {
				return fmt.Errorf("input tensors not on the same device")
			}

		case *cudatensor.CUDATensor:
			if dev == 0 {
				dev = core.CUDA
			} else if dev != core.CUDA {
				return fmt.Errorf("input tensors not on the same device")
			}

		default:
			return fmt.Errorf("unsupported tensor implementation")
		}
	}

	return nil
}
