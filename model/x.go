package model

import (
	qca "github.com/sahandsafizadeh/qeep/component/activation"
	qcw "github.com/sahandsafizadeh/qeep/component/weighted"
	"github.com/sahandsafizadeh/qeep/model/layers"
)

func testFunc() {
	x := layers.Input()

	x = layers.FC(&qcw.FCConfig{Outputs: 64})(x)
	x = layers.Tanh()(x)

	x = layers.FC(&qcw.FCConfig{Outputs: 32})(x)
	x = layers.Sigmoid()(x)

	x = layers.FC(&qcw.FCConfig{Outputs: 16})(x)
	x = layers.Relu()(x)

	x = layers.FC(&qcw.FCConfig{Outputs: 8})(x)
	x = layers.LeakyRelu(nil)(x)

	x = layers.FC(&qcw.FCConfig{Outputs: 4})(x)
	x = layers.Softmax(&qca.SoftmaxConfig{Dim: 1})(x)

	_ = x
}
