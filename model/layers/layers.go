package layers

import (
	qc "github.com/sahandsafizadeh/qeep/component"
	qca "github.com/sahandsafizadeh/qeep/component/activation"
	qcm "github.com/sahandsafizadeh/qeep/component/module"
	"github.com/sahandsafizadeh/qeep/model/node"
)

func Tanh() node.StreamFunc1 {
	return node.NextStreamFunc1(func() (qc.Component, error) {
		return qca.NewTanh(), nil
	})
}

func Sigmoid() node.StreamFunc1 {
	return node.NextStreamFunc1(func() (qc.Component, error) {
		return qca.NewSigmoid(), nil
	})
}

func Softmax(conf *qca.SoftmaxConfig) node.StreamFunc1 {
	return node.NextStreamFunc1(func() (qc.Component, error) {
		return qca.NewSoftmax(conf)
	})
}

func Relu() node.StreamFunc1 {
	return node.NextStreamFunc1(func() (qc.Component, error) {
		return qca.NewRelu(), nil
	})
}

func LeakyRelu(conf *qca.LeakyReluConfig) node.StreamFunc1 {
	return node.NextStreamFunc1(func() (qc.Component, error) {
		return qca.NewLeakyRelu(conf), nil
	})
}

func FC(conf *qcm.FCConfig) node.StreamFunc1 {
	return node.NextStreamFunc1(func() (qc.Component, error) {
		return qcm.NewFC(conf)
	})
}
