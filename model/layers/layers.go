package layers

import (
	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/model/internal/types"
	"github.com/sahandsafizadeh/qeep/model/stream"
)

func Input() *stream.Stream {
	initf := func() (types.Layer, error) {
		return layers.NewInput(), nil
	}

	return stream.NewStream(initf, nil)
}

func Tanh() stream.Func {
	initf := func() (types.Layer, error) {
		return activations.NewTanh(), nil
	}

	return stream.NewStreamFunc(initf)
}

func Sigmoid() stream.Func {
	initf := func() (types.Layer, error) {
		return activations.NewSigmoid(), nil
	}

	return stream.NewStreamFunc(initf)
}

func Softmax(conf *activations.SoftmaxConfig) stream.Func {
	initf := func() (types.Layer, error) {
		return activations.NewSoftmax(conf)
	}

	return stream.NewStreamFunc(initf)
}

func Relu() stream.Func {
	initf := func() (types.Layer, error) {
		return activations.NewRelu(), nil
	}

	return stream.NewStreamFunc(initf)
}

func LeakyRelu(conf *activations.LeakyReluConfig) stream.Func {
	initf := func() (types.Layer, error) {
		return activations.NewLeakyRelu(conf), nil
	}

	return stream.NewStreamFunc(initf)
}

func FC(conf *layers.FCConfig) stream.Func {
	initf := func() (types.Layer, error) {
		return layers.NewFC(conf)
	}

	return stream.NewStreamFunc(initf)
}
