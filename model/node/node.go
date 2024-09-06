package node

import qt "github.com/sahandsafizadeh/qeep/tensor"

func (n *NodeBase) Output() (o qt.Tensor) {
	return n.output
}

func (n *NodeBase) SetOutput(o qt.Tensor) {
	n.output = o
}

func (n *NodeBase) AddChild(c Node) {
	n.children = append(n.children, c)
}
