package node

import (
	qc "github.com/sahandsafizadeh/qeep/component"
	qt "github.com/sahandsafizadeh/qeep/tensor"
)

func NewNode(compInitFunc componentInitializerFunc) (n *Node, err error) {
	comp, err := compInitFunc()
	if err != nil {
		return
	}

	return &Node{
		component: comp,
	}, nil
}

func (n *Node) AddParent(c *Node) {
	n.parents = append(n.parents, c)
}

func (n *Node) AddChild(c *Node) {
	n.children = append(n.children, c)
}

func (n *Node) Children() (children []*Node) {
	return n.children
}

func (n *Node) Result() (o qt.Tensor) {
	return n.result
}

func (n *Node) Forward() (err error) {
	xs := make([]qt.Tensor, len(n.parents))
	for i, p := range n.parents {
		xs[i] = p.Result()
	}

	y, err := n.component.Forward(xs...)
	if err != nil {
		return err
	}

	n.result = y

	return nil
}

func (n *Node) Optimize(optimFunc qc.OptimizerFunc) (err error) {
	wComp, ok := n.component.(qc.WeightedComponent)
	if !ok {
		return nil
	}

	for _, w := range wComp.Weights() {
		err = optimFunc(w)
		if err != nil {
			return
		}
	}

	return nil
}
