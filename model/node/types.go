package node

import (
	qc "github.com/sahandsafizadeh/qeep/component"
	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type Node interface {
	AddChild(Node)
	Children() []Node
	Output() qt.Tensor
	Forward() error
	Optimize(qc.OptimizerFunc) error
}

type ModelNode struct {
	parents   []Node
	children  []Node
	component qc.Component
	output    qt.Tensor
}
