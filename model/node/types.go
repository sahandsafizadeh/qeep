package node

import qt "github.com/sahandsafizadeh/qeep/tensor"

type Node interface {
	/*--------------- implement --------------*/
	Forward() (qt.Tensor, error)

	/*--------------- built in ---------------*/
	Output() qt.Tensor
	SetOutput(qt.Tensor)
	AddChild(Node)
}

type NodeBase struct {
	output   qt.Tensor
	children []Node
}
