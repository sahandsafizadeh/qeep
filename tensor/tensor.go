// Tensor is a data structure with a variable number of dimensions, supporting statistical and linear algebra operations.
// As the heart of the qeep project, many design decisions such as the use of build tags, cgo, and device abstraction are driven by this package.
//
// By design:
// Tensor is defined as an interface, allowing for different internal implementations across various devices (e.g., CPU, CUDA).
// Tensor implementations must be immutable.
// Tensor implementations must support automatic gradient computation using the gradtrack package.
// Every tensor must maintain a valid gradient state (i.e., a GradContext), which is the only mutable part of a tensor.
// After backpropagation, the GradContext becomes invalid and must be explicitly reset.
package tensor

import (
	"github.com/sahandsafizadeh/qeep/tensor/internal/core"
	"github.com/sahandsafizadeh/qeep/tensor/internal/impl/dispatch"
)

// Tensor is a multi-dimensional array supporting linear algebra and automatic differentiation.
type Tensor = core.Tensor

// Config carries tensor initializer options: target device and gradient tracking.
type Config = core.Config

// Device selects where tensor data and computation live.
type Device = core.Device

// Range specifies a half-open interval [Start, End) for slicing.
type Range = core.Range

const (
	CPU  = core.CPU
	CUDA = core.CUDA
)

// Full returns a tensor with the given shape, where every element equals value.
func Full(dims []int, value float64, conf *Config) (Tensor, error) {
	return dispatch.Full(dims, value, conf)
}

// Zeros returns a tensor with the given shape, filled with zeros.
func Zeros(dims []int, conf *Config) (Tensor, error) {
	return dispatch.Zeros(dims, conf)
}

// Ones returns a tensor with the given shape, filled with ones.
func Ones(dims []int, conf *Config) (Tensor, error) {
	return dispatch.Ones(dims, conf)
}

// Eye returns a d-by-d identity matrix (ones on diagonal, zeros elsewhere).
func Eye(d int, conf *Config) (Tensor, error) {
	return dispatch.Eye(d, conf)
}

// RandU returns a tensor with the given shape, filled with uniformly distributed random values in [l, u).
func RandU(dims []int, l, u float64, conf *Config) (Tensor, error) {
	return dispatch.RandU(dims, l, u, conf)
}

// RandN returns a tensor with shape dims filled with normally distributed random values.
func RandN(dims []int, u, s float64, conf *Config) (Tensor, error) {
	return dispatch.RandN(dims, u, s, conf)
}

// Of returns a tensor from slice (float64 or nested slices up to 4D).
func Of[T core.InputDataType](data T, conf *Config) (Tensor, error) {
	return dispatch.Of(data, conf)
}

// Load returns a tensor restored from the file written by Save to path.
func Load(path string, conf *Config) (Tensor, error) {
	return dispatch.Load(path, conf)
}

// Transfer returns a tensor placed on the given device.
func Transfer(t Tensor, to Device) (Tensor, error) {
	return dispatch.Transfer(t, to)
}

// Concat joins tensors along the specified dimension.
// All input tensors must reside on the same device and have compatible shapes.
func Concat(ts []Tensor, dim int) (Tensor, error) {
	return dispatch.Concat(ts, dim)
}

// BackPropagate computes gradients for t and all tensors in its computation graph.
// After backpropagation, gradient contexts become invalid and must be reset before reuse.
func BackPropagate(t Tensor) error {
	return dispatch.BackPropagate(t)
}

// Save writes tensor's elements and shape to path. Gradient state is not stored.
func Save(t Tensor, path string) error {
	return dispatch.Save(t, path)
}

// RunTestLogicOnDevices is a test helper that runs testLogic on every available device.
func RunTestLogicOnDevices(testLogic func(Device)) {
	dispatch.RunTestLogicOnDevices(testLogic)
}
