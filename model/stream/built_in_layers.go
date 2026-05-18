package stream

import (
	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/model/internal/contract"
)

func Input() *Stream {
	initf := func() (contract.Layer, error) {
		return layers.NewInput(), nil
	}

	return NewStream(initf, nil)
}

func Tanh() StreamFunc {
	initf := func() (contract.Layer, error) {
		return activations.NewTanh(), nil
	}

	return NewStreamFunc(initf)
}

func Sigmoid() StreamFunc {
	initf := func() (contract.Layer, error) {
		return activations.NewSigmoid(), nil
	}

	return NewStreamFunc(initf)
}

func Softmax(conf *activations.SoftmaxConfig) StreamFunc {
	initf := func() (contract.Layer, error) {
		return activations.NewSoftmax(conf)
	}

	return NewStreamFunc(initf)
}

func Relu() StreamFunc {
	initf := func() (contract.Layer, error) {
		return activations.NewRelu(), nil
	}

	return NewStreamFunc(initf)
}

func LeakyRelu(conf *activations.LeakyReluConfig) StreamFunc {
	initf := func() (contract.Layer, error) {
		return activations.NewLeakyRelu(conf), nil
	}

	return NewStreamFunc(initf)
}

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

func FC(conf *layers.FCConfig) StreamFunc {
	initf := func() (contract.Layer, error) {
		return layers.NewFC(conf)
	}

	return NewStreamFunc(initf)
}
