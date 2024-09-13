package model

import (
	"github.com/sahandsafizadeh/qeep/model/layers"
	"github.com/sahandsafizadeh/qeep/model/node"
)

func testFunc() {
	// x := Input(64)
	x := new(node.Stream)

	// first layer
	// x = FC(32)(x)
	x = layers.Relu()(x)

	// second layer
	// x = FC(16)(x)
	x = layers.Tanh()(x)

	// output layer
	// x = FC(1)(x)
	x = layers.Sigmoid()(x)

	_ = x
}
