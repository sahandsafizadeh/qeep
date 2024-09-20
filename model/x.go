package model

import (
	qca "github.com/sahandsafizadeh/qeep/component/activation"
	qcm "github.com/sahandsafizadeh/qeep/component/module"
	"github.com/sahandsafizadeh/qeep/model/layers"
)

func testFunc() {
	x := layers.Input()

	x = layers.FC(&qcm.FCConfig{N: 64})(x)
	x = layers.Tanh()(x)

	x = layers.FC(&qcm.FCConfig{N: 32})(x)
	x = layers.Sigmoid()(x)

	x = layers.FC(&qcm.FCConfig{N: 16})(x)
	x = layers.Relu()(x)

	x = layers.FC(&qcm.FCConfig{N: 8})(x)
	x = layers.LeakyRelu(nil)(x)

	x = layers.FC(&qcm.FCConfig{N: 4})(x)
	x = layers.Softmax(&qca.SoftmaxConfig{Dim: 1})(x)

	_ = x
}
