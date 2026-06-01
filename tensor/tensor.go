package tensor

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/cputensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/cudatensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
)

// Full returns a tensor with the given shape, where every element equals value.
func Full(dims []int, value float64, conf *Config) (t tensor.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("Full tensor config data validation failed: %w", err)
	}

	switch conf.Device {
	case CPU:
		t, err = cputensor.Full(dims, value, conf.GradTrack)
	case CUDA:
		t, err = cudatensor.Full(dims, value, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

// Zeros returns a tensor with the given shape, filled with zeros.
func Zeros(dims []int, conf *Config) (t tensor.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("Zeros tensor config data validation failed: %w", err)
	}

	switch conf.Device {
	case CPU:
		t, err = cputensor.Zeros(dims, conf.GradTrack)
	case CUDA:
		t, err = cudatensor.Zeros(dims, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

// Ones returns a tensor with the given shape, filled with ones.
func Ones(dims []int, conf *Config) (t tensor.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("Ones tensor config data validation failed: %w", err)
	}

	switch conf.Device {
	case CPU:
		t, err = cputensor.Ones(dims, conf.GradTrack)
	case CUDA:
		t, err = cudatensor.Ones(dims, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

// Eye returns a d-by-d identity matrix (ones on diagonal, zeros elsewhere).
func Eye(d int, conf *Config) (t tensor.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("Eye tensor config data validation failed: %w", err)
	}

	switch conf.Device {
	case CPU:
		t, err = cputensor.Eye(d, conf.GradTrack)
	case CUDA:
		t, err = cudatensor.Eye(d, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

// RandU returns a tensor with the given shape, filled with uniformly distributed random values in [l, u).
func RandU(dims []int, l, u float64, conf *Config) (t tensor.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("RandU tensor config data validation failed: %w", err)
	}

	switch conf.Device {
	case CPU:
		t, err = cputensor.RandU(dims, l, u, conf.GradTrack)
	case CUDA:
		t, err = cudatensor.RandU(dims, l, u, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

// RandN returns a tensor with shape dims filled with normally distributed random values.
func RandN(dims []int, u, s float64, conf *Config) (t tensor.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("RandN tensor config data validation failed: %w", err)
	}

	switch conf.Device {
	case CPU:
		t, err = cputensor.RandN(dims, u, s, conf.GradTrack)
	case CUDA:
		t, err = cudatensor.RandN(dims, u, s, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

// Of creates a tensor from slice (float64 or nested slices up to 4D).
func Of[T inputDataType](data T, conf *Config) (t tensor.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("Of tensor config data validation failed: %w", err)
	}

	switch conf.Device {
	case CPU:
		t, err = cputensor.Of(data, conf.GradTrack)
	case CUDA:
		t, err = cudatensor.Of(data, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

// Concat joins tensors along the specified dimension.
// All input tensors must reside on the same device and have compatible shapes.
func Concat(ts []tensor.Tensor, dim int) (t tensor.Tensor, err error) {
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

// BackPropagate computes gradients for t and all tensors in its computation graph.
// After backpropagation, gradient contexts become invalid and must be reset before reuse.
func BackPropagate(t tensor.Tensor) (err error) {
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
	case CUDA:
	default:
		return conf, fmt.Errorf("invalid input device")
	}

	return conf, nil
}

func validateImplementation(t tensor.Tensor) (err error) {
	switch t.(type) {
	case *cputensor.CPUTensor,
		*cudatensor.CUDATensor:
		return nil

	default:
		return fmt.Errorf("unsupported tensor implementation")
	}
}

func validateImplementationsUnity(ts []tensor.Tensor) (err error) {
	if len(ts) < 2 {
		return fmt.Errorf("expected at least (2) tensors: got (%d)", len(ts))
	}

	var dev Device

	for _, t := range ts {
		switch t.(type) {
		case *cputensor.CPUTensor:
			if dev == 0 {
				dev = CPU
			} else if dev != CPU {
				return fmt.Errorf("input tensors not on the same device")
			}

		case *cudatensor.CUDATensor:
			if dev == 0 {
				dev = CUDA
			} else if dev != CUDA {
				return fmt.Errorf("input tensors not on the same device")
			}

		default:
			return fmt.Errorf("unsupported tensor implementation")
		}
	}

	return nil
}

/* ----- testing helpers ----- */

func RunTestLogicOnDevices(testLogic func(Device)) {
	devices := []Device{CPU}

	if cudatensor.IsAvailable {
		devices = append(devices, CUDA)
	}

	for _, dev := range devices {
		testLogic(dev)
	}
}
