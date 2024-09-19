package model

import (
	qca "github.com/sahandsafizadeh/qeep/component/activation"
	qcm "github.com/sahandsafizadeh/qeep/component/module"
	"github.com/sahandsafizadeh/qeep/model/layers"
	"github.com/sahandsafizadeh/qeep/model/node"
)

func testFunc() {
	// x := Input(128)
	x := new(node.Stream)

	// Layer 1
	x = layers.FC(&qcm.FCConfig{N: 64})(x)
	x = layers.Tanh()(x)

	// Layer 2
	x = layers.FC(&qcm.FCConfig{N: 32})(x)
	x = layers.Sigmoid()(x)

	// Layer 3
	x = layers.FC(&qcm.FCConfig{N: 16})(x)
	x = layers.Relu()(x)

	// Layer 4
	x = layers.FC(&qcm.FCConfig{N: 8})(x)
	x = layers.LeakyRelu(nil)(x)

	// Layer Output
	x = layers.FC(&qcm.FCConfig{N: 4})(x)
	x = layers.Softmax(&qca.SoftmaxConfig{Dim: 1})(x)

	_ = x
}
