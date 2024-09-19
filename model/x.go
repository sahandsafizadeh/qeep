package model

import (
	qca "github.com/sahandsafizadeh/qeep/component/activation"
	"github.com/sahandsafizadeh/qeep/model/layers"
	"github.com/sahandsafizadeh/qeep/model/node"
)

func testFunc() {
	// x := Input(128)
	x := new(node.Stream)

	// Layer 1
	// x = FC(64)(x)
	x = layers.Tanh()(x)

	// Layer 2
	// x = FC(32)(x)
	x = layers.Sigmoid()(x)

	// Layer 3
	// x = FC(16)(x)
	x = layers.Relu()(x)

	// Layer 4
	// x = FC(8)(x)
	x = layers.LeakyRelu(nil)(x)

	// Layer Output
	// x = FC(4)(x)
	x = layers.Softmax(&qca.SoftmaxConfig{Dim: 1})(x)

	_ = x
}
