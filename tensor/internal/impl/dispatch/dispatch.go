package dispatch

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/core"
	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
	"github.com/sahandsafizadeh/qeep/tensor/internal/impl/cputensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/impl/cudatensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/persist"
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

func Of[T core.InputDataType](data T, conf *core.Config) (t core.Tensor, err error) {
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

func Load(path string, conf *core.Config) (t core.Tensor, err error) {
	conf, err = toValidConfig(conf)
	if err != nil {
		return t, fmt.Errorf("Load tensor config data validation failed: %w", err)
	}

	snapshot, err := persist.Load(path)
	if err != nil {
		return t, fmt.Errorf("Load operation failed: %w", err)
	}

	switch conf.Device {
	case core.CPU:
		t, err = cputensor.Import(snapshot, conf.GradTrack)
	case core.CUDA:
		t, err = cudatensor.Import(snapshot, conf.GradTrack)
	default:
		panic("unreachable: unsupported device")
	}

	if err != nil {
		return t, fmt.Errorf("%s initialization: %w", conf.Device, err)
	}

	return t, nil
}

func Transfer(t core.Tensor, to core.Device) (o core.Tensor, err error) {
	err = validateImplementation(t)
	if err != nil {
		return o, fmt.Errorf("Transfer tensor implementation validation failed: %w", err)
	}

	if t.Device() == to {
		return t, nil
	}

	expt := t.(core.ExporterTensor)

	switch to {
	case core.CPU:
		o, err = cputensor.Transfer(expt)
	case core.CUDA:
		o, err = cudatensor.Transfer(expt)
	default:
		return o, fmt.Errorf("Transfer target device validation failed: invalid input device")
	}

	if err != nil {
		return o, fmt.Errorf("%s initialization: %w", to, err)
	}

	return o, nil
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

func Save(t core.Tensor, path string) (err error) {
	err = validateImplementation(t)
	if err != nil {
		return fmt.Errorf("Save tensor implementation validation failed: %w", err)
	}

	snapshot := t.(core.ExporterTensor).Export()

	err = persist.Save(snapshot, path)
	if err != nil {
		return fmt.Errorf("Save operation failed: %w", err)
	}

	return nil
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

func ResetGradient(t core.Tensor, tracked bool) (err error) {
	err = validateImplementation(t)
	if err != nil {
		return fmt.Errorf("ResetGradient tensor implementation validation failed: %w", err)
	}

	gradtrack.ResetGradient(t, tracked)

	return nil
}

func RunTestLogicOnDevices(testLogic func(core.Device)) {
	devices := []core.Device{core.CPU}

	if cudatensor.IsAvailable {
		devices = append(devices, core.CUDA)
	}

	for _, dev := range devices {
		testLogic(dev)
	}
}

func RunTestLogicCrossDevice(testLogic func(core.Device, core.Device)) {
	type deviceTuple struct {
		dev1 core.Device
		dev2 core.Device
	}

	devtups := make([]deviceTuple, 0)

	if cudatensor.IsAvailable {
		devtups = append(devtups, deviceTuple{core.CPU, core.CUDA})
		devtups = append(devtups, deviceTuple{core.CUDA, core.CPU})
	}

	for _, tup := range devtups {
		testLogic(tup.dev1, tup.dev2)
	}
}
