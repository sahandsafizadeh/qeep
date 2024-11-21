package tinit

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/cputensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
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

func validateTensorDevice(t tensor.Tensor) (err error) {
	switch t.(type) {
	case *cputensor.CPUTensor:
		return nil

	case nil:
		err = fmt.Errorf("expected input tensor not to be nil")
		return

	default:
		err = fmt.Errorf("unsupported tensor implementation")
		return
	}
}

func validateTensorsDeviceUnity(ts []tensor.Tensor) (err error) {
	if len(ts) < 2 {
		err = fmt.Errorf("expected at least (2) tensors: got (%d)", len(ts))
		return
	}

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

		case nil:
			err = fmt.Errorf("expected input tensor not to be nil")
			return

		default:
			err = fmt.Errorf("unsupported tensor implementation")
			return
		}
	}

	return nil
}
