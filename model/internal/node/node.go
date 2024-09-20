package node

import (
	"fmt"

	qc "github.com/sahandsafizadeh/qeep/component"
	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type Node struct {
	component qc.Component
	result    qt.Tensor
	parents   []*Node
	children  []*Node
}

type ComponentInitializerFunc func() (qc.Component, error)

func NewNode(compInitFunc ComponentInitializerFunc) (n *Node, err error) {
	comp, err := compInitFunc()
	if err != nil {
		return
	} else if comp == nil {
		err = fmt.Errorf("expected compInitFunc not to return nil")
		return
	}

	return &Node{
		component: comp,
	}, nil
}

func (n *Node) Result() (result qt.Tensor) {
	return n.result
}

func (n *Node) Parents() (parents []*Node) {
	parents = make([]*Node, len(n.parents))
	copy(parents, n.parents)
	return parents
}

func (n *Node) Children() (children []*Node) {
	children = make([]*Node, len(n.children))
	copy(children, n.children)
	return children
}

func (n *Node) AddParent(p *Node) (err error) {
	if p == nil {
		err = fmt.Errorf("expected parent not to be nil")
		return
	}

	n.parents = append(n.parents, p)

	return nil
}

func (n *Node) AddChild(c *Node) (err error) {
	if c == nil {
		err = fmt.Errorf("expected child not to be nil")
		return
	}

	n.children = append(n.children, c)

	return nil
}

func (n *Node) Forward() (err error) {
	xs := make([]qt.Tensor, len(n.parents))
	for i, p := range n.parents {
		xs[i] = p.result
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
