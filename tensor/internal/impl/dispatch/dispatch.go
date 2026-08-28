package dispatch

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/core"
	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
	"github.com/sahandsafizadeh/qeep/tensor/internal/impl/cputensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/impl/cudatensor"
)

func Full(dims []int, value float64, conf *core.Config) (t core.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("Full tensor config data validation failed: %w", err)
	}

	switch conf.Device {
	case core.CPU:
		t, err = cputensor.Full(dims, value, conf.GradTrack)
	case core.CUDA:
		t, err = cudatensor.Full(dims, value, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

func Zeros(dims []int, conf *core.Config) (t core.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("Zeros tensor config data validation failed: %w", err)
	}

	switch conf.Device {
	case core.CPU:
		t, err = cputensor.Zeros(dims, conf.GradTrack)
	case core.CUDA:
		t, err = cudatensor.Zeros(dims, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

func Ones(dims []int, conf *core.Config) (t core.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("Ones tensor config data validation failed: %w", err)
	}

	switch conf.Device {
	case core.CPU:
		t, err = cputensor.Ones(dims, conf.GradTrack)
	case core.CUDA:
		t, err = cudatensor.Ones(dims, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

func Eye(d int, conf *core.Config) (t core.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("Eye tensor config data validation failed: %w", err)
	}

	switch conf.Device {
	case core.CPU:
		t, err = cputensor.Eye(d, conf.GradTrack)
	case core.CUDA:
		t, err = cudatensor.Eye(d, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

func RandU(dims []int, l, u float64, conf *core.Config) (t core.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("RandU tensor config data validation failed: %w", err)
	}

	switch conf.Device {
	case core.CPU:
		t, err = cputensor.RandU(dims, l, u, conf.GradTrack)
	case core.CUDA:
		t, err = cudatensor.RandU(dims, l, u, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

func RandN(dims []int, u, s float64, conf *core.Config) (t core.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("RandN tensor config data validation failed: %w", err)
	}

	switch conf.Device {
	case core.CPU:
		t, err = cputensor.RandN(dims, u, s, conf.GradTrack)
	case core.CUDA:
		t, err = cudatensor.RandN(dims, u, s, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

func Of[T InputDataType](data T, conf *core.Config) (t core.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("Of tensor config data validation failed: %w", err)
	}

	switch conf.Device {
	case core.CPU:
		t, err = cputensor.Of(data, conf.GradTrack)
	case core.CUDA:
		t, err = cudatensor.Of(data, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

func Concat(ts []core.Tensor, dim int) (t core.Tensor, err error) {
	err = validateImplementationsUnity(ts)
	if err != nil {
		return t, fmt.Errorf("Concat tensor implementation validation failed: %w", err)
	}

	switch ts[0].(type) {
	case *cputensor.CPUTensor:
		t, err = cputensor.Concat(ts, dim)
	case *cudatensor.CUDATensor:
		t, err = cudatensor.Concat(ts, dim)
	default:
		panic("unreachable: unsupported implementation")
	}

	if err != nil {
		return t, fmt.Errorf("Concat: %w", err)
	}

	return t, nil
}

func BackPropagate(t core.Tensor) (err error) {
	err = validateImplementation(t)
	if err != nil {
		return fmt.Errorf("BackPropagate tensor implementation validation failed: %w", err)
	}

	err = gradtrack.BackPropagate(t)
	if err != nil {
		return fmt.Errorf("BackPropagate operation failed: %w", err)
	}

	return nil
}

/* ----- helpers ----- */

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
