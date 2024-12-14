package tensor

import (
	"fmt"
	"strconv"

	"github.com/sahandsafizadeh/qeep/tensor/internal/cputensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
)

func Full(dims []int, value float64, conf *Config) (t tensor.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		err = fmt.Errorf("Full tensor config data validation failed: %w", err)
		return
	}

	switch conf.Device {
	case CPU:
		return cputensor.Full(dims, value, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}
}

func Zeros(dims []int, conf *Config) (t tensor.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		err = fmt.Errorf("Zeros tensor config data validation failed: %w", err)
		return
	}

	switch conf.Device {
	case CPU:
		return cputensor.Zeros(dims, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}
}

func Ones(dims []int, conf *Config) (t tensor.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		err = fmt.Errorf("Ones tensor config data validation failed: %w", err)
		return
	}

	switch conf.Device {
	case CPU:
		return cputensor.Ones(dims, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}
}

func Eye(n int, conf *Config) (t tensor.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		err = fmt.Errorf("Eye tensor config data validation failed: %w", err)
		return
	}

	switch conf.Device {
	case CPU:
		return cputensor.Eye(n, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}
}

func RandU(dims []int, l, u float64, conf *Config) (t tensor.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		err = fmt.Errorf("RandU tensor config data validation failed: %w", err)
		return
	}

	switch conf.Device {
	case CPU:
		return cputensor.RandU(dims, l, u, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}
}

func RandN(dims []int, u, s float64, conf *Config) (t tensor.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		err = fmt.Errorf("RandN tensor config data validation failed: %w", err)
		return
	}

	switch conf.Device {
	case CPU:
		return cputensor.RandN(dims, u, s, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}
}

func TensorOf[T inputDataType](data T, conf *Config) (t tensor.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		err = fmt.Errorf("TensorOf tensor config data validation failed: %w", err)
		return
	}

	switch conf.Device {
	case CPU:
		return cputensor.TensorOf(data, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}
}

func Concat(ts []tensor.Tensor, dim int) (o tensor.Tensor, err error) {
	err = validateImplementationsUnity(ts)
	if err != nil {
		err = fmt.Errorf("Concat tensor implementation validation failed: %w", err)
		return
	}

	switch ts[0].(type) {
	case *cputensor.CPUTensor:
		return cputensor.Concat(ts, dim)
	default:
		panic("unreachable: unsupported implementation")
	}
}

func BackPropagate(t tensor.Tensor) (err error) {
	err = validateImplementation(t)
	if err != nil {
		err = fmt.Errorf("BackPropagate tensor implementation validation failed: %w", err)
		return
	}

	return gradtrack.BackPropagate(t)
}

/* ----- helpers ----- */

func toValidConfig(iconf *Config) (conf *Config, err error) {
	if iconf == nil {
		iconf = &Config{
			Device:    CPU,
			GradTrack: false,
		}
	}

	conf = new(Config)
	*conf = *iconf

	switch conf.Device {
	case CPU:
	default:
		err = fmt.Errorf("invalid input device")
		return
	}

	return conf, nil
}

func validateImplementation(t tensor.Tensor) (err error) {
	switch t.(type) {
	case *cputensor.CPUTensor:
		return nil

	default:
		err = fmt.Errorf("unsupported tensor implementation")
		return
	}
}

func validateImplementationsUnity(ts []tensor.Tensor) (err error) {
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

		default:
			err = fmt.Errorf("unsupported tensor implementation")
			return
		}
	}

	return nil
}

/* ----- testing helpers ----- */

func RunTestLogicOnDevices(testLogic func(Device)) {
	devices := []Device{CPU}
	for _, dev := range devices {
		testLogic(dev)
	}
}

func (d Device) String() string {
	switch d {
	case CPU:
		return "CPU"
	default:
		return strconv.Itoa(int(d))
	}
}
