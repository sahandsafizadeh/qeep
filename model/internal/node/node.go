package node

import (
	qc "github.com/sahandsafizadeh/qeep/component"
	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type Node struct {
	isInput   bool
	parents   []*Node
	children  []*Node
	component qc.Component
	result    qt.Tensor
}

type ComponentInitializerFunc func() (qc.Component, error)

func NewInputNode() (n *Node) {
	return &Node{
		isInput: true,
	}
}

func NewCompNode(compInitFunc ComponentInitializerFunc) (n *Node, err error) {
	comp, err := compInitFunc()
	if err != nil {
		return
	}

	return &Node{
		isInput:   false,
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

func (n *Node) SetResult(r qt.Tensor) {
	n.result = r
}

func (n *Node) Result() (r qt.Tensor) {
	return n.result
}

func (n *Node) Forward() (err error) {
	if n.isInput {
		return nil
	}

	xs := make([]qt.Tensor, len(n.parents))
	for i, p := range n.parents {
		xs[i] = p.Result()
	}

	y, err := n.component.Forward(xs...)
	if err != nil {
		return err
	}

	n.SetResult(y)

	return nil
}

func (n *Node) Optimize(optimFunc qc.OptimizerFunc) (err error) {
	if n.isInput {
		return nil
	}

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
