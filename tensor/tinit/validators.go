package tinit

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/cputensor"
)

func validateConfig(conf *Config) (err error) {
	if conf == nil {
		return nil
	}

	switch conf.Device {
	case CPU:
	default:
		err = fmt.Errorf("invalid input device")
		return
	}

	return nil
}

func validateTensorsDeviceUnity(ts []tensor.Tensor) (err error) {
	var dev Device

	for _, t := range ts {
		switch t.(type) {
		case *cputensor.CPUTensor:
			if dev == 0 {
				dev = CPU
			} else if dev != CPU {
				err = fmt.Errorf("input tensors not on the same device")
				return
			}

		default:
			err = fmt.Errorf("unsupported tensor implementation")
			return
		}
	}

	if dev == 0 {
		err = fmt.Errorf("expected at least (2) tensors: got (0)")
		return
	}

	return nil
}
