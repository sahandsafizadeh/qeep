package tinit

import (
	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/cputensor"
)

func Full(dims []int32, value float64, conf *Config) (t qt.Tensor, err error) {
	c, err := prepareConfig(conf)
	if err != nil {
		return
	}

	switch c.Device {
	case CPU:
		return cputensor.Full(dims, value, c.GradTrack)
	default:
		panic("unreachable: unsupported device validated")
	}
}

func Zeros(dims []int32, conf *Config) (t qt.Tensor, err error) {
	c, err := prepareConfig(conf)
	if err != nil {
		return
	}

	switch c.Device {
	case CPU:
		return cputensor.Zeros(dims, c.GradTrack)
	default:
		panic("unreachable: unsupported device validated")
	}
}

func Ones(dims []int32, conf *Config) (t qt.Tensor, err error) {
	c, err := prepareConfig(conf)
	if err != nil {
		return
	}

	switch c.Device {
	case CPU:
		return cputensor.Ones(dims, c.GradTrack)
	default:
		panic("unreachable: unsupported device validated")
	}
}

func Eye(n int32, conf *Config) (t qt.Tensor, err error) {
	c, err := prepareConfig(conf)
	if err != nil {
		return
	}

	switch c.Device {
	case CPU:
		return cputensor.Eye(n, c.GradTrack)
	default:
		panic("unreachable: unsupported device validated")
	}
}

func RandU(dims []int32, l, u float64, conf *Config) (t qt.Tensor, err error) {
	c, err := prepareConfig(conf)
	if err != nil {
		return
	}

	switch c.Device {
	case CPU:
		return cputensor.RandU(dims, l, u, c.GradTrack)
	default:
		panic("unreachable: unsupported device validated")
	}
}

func RandN(dims []int32, u, s float64, conf *Config) (t qt.Tensor, err error) {
	c, err := prepareConfig(conf)
	if err != nil {
		return
	}

	switch c.Device {
	case CPU:
		return cputensor.RandN(dims, u, s, c.GradTrack)
	default:
		panic("unreachable: unsupported device validated")
	}
}

func TensorOf[T inputDataType](data T, conf *Config) (t qt.Tensor, err error) {
	c, err := prepareConfig(conf)
	if err != nil {
		return
	}

	switch c.Device {
	case CPU:
		return cputensor.TensorOf(data, c.GradTrack)
	default:
		panic("unreachable: unsupported device validated")
	}
}

func Concat(ts []qt.Tensor, dim int32) (o qt.Tensor, err error) {
	err = validateTensorsDeviceUnity(ts)
	if err != nil {
		return
	}

	switch ts[0].(type) {
	case *cputensor.CPUTensor:
		return cputensor.Concat(ts, dim)
	default:
		panic("unreachable: unsupported device validated")
	}
}

func prepareConfig(conf *Config) (c Config, err error) {
	err = validateConfig(conf)
	if err != nil {
		return
	}

	if conf == nil {
		return Config{
			Device:    CPU,
			GradTrack: false,
		}, nil
	}

	return *conf, nil
}
