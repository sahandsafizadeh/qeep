package layers

import (
	qc "github.com/sahandsafizadeh/qeep/component"
	qca "github.com/sahandsafizadeh/qeep/component/activation"
	qcw "github.com/sahandsafizadeh/qeep/component/weighted"
	"github.com/sahandsafizadeh/qeep/model/stream"
)

func Input() *stream.Stream {
	return stream.NewStream(func() (qc.Forwarder, error) {
		return NewInput(), nil
	}, nil)
}

func Tanh() stream.StreamFunc1 {
	return stream.NextStreamFunc1(func() (qc.Forwarder, error) {
		return qca.NewTanh(), nil
	})
}

func Sigmoid() stream.StreamFunc1 {
	return stream.NextStreamFunc1(func() (qc.Forwarder, error) {
		return qca.NewSigmoid(), nil
	})
}

func Softmax(conf *qca.SoftmaxConfig) stream.StreamFunc1 {
	return stream.NextStreamFunc1(func() (qc.Forwarder, error) {
		return qca.NewSoftmax(conf)
	})
}

func Relu() stream.StreamFunc1 {
	return stream.NextStreamFunc1(func() (qc.Forwarder, error) {
		return qca.NewRelu(), nil
	})
}

func LeakyRelu(conf *qca.LeakyReluConfig) stream.StreamFunc1 {
	return stream.NextStreamFunc1(func() (qc.Forwarder, error) {
		return qca.NewLeakyRelu(conf), nil
	})
}

func FC(conf *qcw.FCConfig) stream.StreamFunc1 {
	return stream.NextStreamFunc1(func() (qc.Forwarder, error) {
		return qcw.NewFC(conf)
	})
}
