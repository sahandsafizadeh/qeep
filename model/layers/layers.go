package layers

import (
	qc "github.com/sahandsafizadeh/qeep/component"
	qca "github.com/sahandsafizadeh/qeep/component/activation"
	qcm "github.com/sahandsafizadeh/qeep/component/module"
	"github.com/sahandsafizadeh/qeep/model/stream"
)

func Input() *stream.Stream {
	return stream.NewStream(func() (qc.Component, error) {
		return qca.NewInput(), nil
	}, nil)
}

func Tanh() stream.StreamFunc1 {
	return stream.NextStreamFunc1(func() (qc.Component, error) {
		return qca.NewTanh(), nil
	})
}

func Sigmoid() stream.StreamFunc1 {
	return stream.NextStreamFunc1(func() (qc.Component, error) {
		return qca.NewSigmoid(), nil
	})
}

func Softmax(conf *qca.SoftmaxConfig) stream.StreamFunc1 {
	return stream.NextStreamFunc1(func() (qc.Component, error) {
		return qca.NewSoftmax(conf)
	})
}

func Relu() stream.StreamFunc1 {
	return stream.NextStreamFunc1(func() (qc.Component, error) {
		return qca.NewRelu(), nil
	})
}

func LeakyRelu(conf *qca.LeakyReluConfig) stream.StreamFunc1 {
	return stream.NextStreamFunc1(func() (qc.Component, error) {
		return qca.NewLeakyRelu(conf), nil
	})
}

func FC(conf *qcm.FCConfig) stream.StreamFunc1 {
	return stream.NextStreamFunc1(func() (qc.Component, error) {
		return qcm.NewFC(conf)
	})
}
