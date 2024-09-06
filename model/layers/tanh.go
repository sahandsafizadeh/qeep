package layers

import (
	"github.com/sahandsafizadeh/qeep/model/node"

	qc "github.com/sahandsafizadeh/qeep/component"
	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type TanhNode struct {
	node.NodeBase
	component *qc.Tanh
	x         node.Node
}

func (n *TanhNode) Forward() (o qt.Tensor, err error) {
	return n.component.Trigger(n.x.Output())
}

func NewTanhNode(x node.Node) *TanhNode {
	return &TanhNode{
		component: qc.NewTanh(),
		x:         x,
	}
}

func Tanh() func(x *Path) (y *Path) {
	return func(x *Path) (y *Path) {
		n := NewTanhNode(x.currentState)
		x.currentState.AddChild(n)
		return &Path{currentState: n}
	}
}
