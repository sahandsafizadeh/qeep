package tinit

import (
	"github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/cputensor"
)

func Full(conf *Config, value float64, dims ...int32) (t tensor.Tensor, err error) {
	c, err := prepareConfig(conf)
	if err != nil {
		return
	}

	switch c.Device {
	case CPU:
		return cputensor.Full(value, dims, c.GradTrack)
	default:
		panic("unreachable: unsupported device validated")
	}
}

func Zeros(conf *Config, dims ...int32) (t tensor.Tensor, err error) {
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

func Ones(conf *Config, dims ...int32) (t tensor.Tensor, err error) {
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

func Eye(conf *Config, n int32) (t tensor.Tensor, err error) {
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

func RandU(conf *Config, l, u float64, dims ...int32) (t tensor.Tensor, err error) {
	c, err := prepareConfig(conf)
	if err != nil {
		return
	}

	switch c.Device {
	case CPU:
		return cputensor.RandU(l, u, dims, c.GradTrack)
	default:
		panic("unreachable: unsupported device validated")
	}
}

func RandN(conf *Config, u, s float64, dims ...int32) (t tensor.Tensor, err error) {
	c, err := prepareConfig(conf)
	if err != nil {
		return
	}

	switch c.Device {
	case CPU:
		return cputensor.RandN(u, s, dims, c.GradTrack)
	default:
		panic("unreachable: unsupported device validated")
	}
}

func TensorOf[T inputDataType](conf *Config, data T) (t tensor.Tensor, err error) {
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

func Concat(ts []tensor.Tensor, dim int32) (o tensor.Tensor, err error) {
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
