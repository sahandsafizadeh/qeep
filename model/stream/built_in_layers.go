package stream

import (
	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/model/internal/contract"
)

// Input returns a stream that acts as a model input. Use it as the root of your network (e.g. x := stream.Input()).
func Input() *Stream {
	initf := func() (contract.Layer, error) {
		return layers.NewInput(), nil
	}

	return NewStream(initf, nil)
}

// Tanh returns a StreamFunc that applies element-wise tanh to its input.
func Tanh() StreamFunc {
	initf := func() (contract.Layer, error) {
		return activations.NewTanh(), nil
	}

	return NewStreamFunc(initf)
}

// Sigmoid returns a StreamFunc that applies element-wise sigmoid to its input.
func Sigmoid() StreamFunc {
	initf := func() (contract.Layer, error) {
		return activations.NewSigmoid(), nil
	}

	return NewStreamFunc(initf)
}

// Softmax returns a StreamFunc that applies softmax along the dimension given in conf (e.g. Dim: 1 for classes).
func Softmax(conf *activations.SoftmaxConfig) StreamFunc {
	initf := func() (contract.Layer, error) {
		return activations.NewSoftmax(conf)
	}

	return NewStreamFunc(initf)
}

// Relu returns a StreamFunc that applies element-wise ReLU (max(0, x)) to its input.
func Relu() StreamFunc {
	initf := func() (contract.Layer, error) {
		return activations.NewRelu(), nil
	}

	return NewStreamFunc(initf)
}

// LeakyRelu returns a StreamFunc that applies Leaky ReLU.
// The negative slope is specified in conf; it determines output for x < 0.
func LeakyRelu(conf *activations.LeakyReluConfig) StreamFunc {
	initf := func() (contract.Layer, error) {
		return activations.NewLeakyRelu(conf), nil
	}

	return NewStreamFunc(initf)
}

// Dropout returns a StreamFunc that applies dropout with the probability from conf (training only).
func Dropout(conf *layers.DropoutConfig) StreamFunc {
	initf := func() (contract.Layer, error) {
		return layers.NewDropout(conf)
	}

	return NewStreamFunc(initf)
}

func BatchNorm(conf *layers.BatchNormConfig) StreamFunc {
	initf := func() (contract.Layer, error) {
		return layers.NewBatchNorm(conf)
	}

	return NewStreamFunc(initf)
}

// FC returns a StreamFunc for a fully connected layer with the given config (inputs, outputs, device, initializers).
func FC(conf *layers.FCConfig) StreamFunc {
	initf := func() (contract.Layer, error) {
		return layers.NewFC(conf)
	}

	return NewStreamFunc(initf)
}
