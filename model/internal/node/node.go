package node

import (
	"fmt"

	qc "github.com/sahandsafizadeh/qeep/component"
	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type Node struct {
	forwarder qc.Forwarder
	result    qt.Tensor
	parents   []*Node
	children  []*Node
}

type ForwarderInitFunc func() (qc.Forwarder, error)

func NewNode(forwarderInitFunc ForwarderInitFunc) (n *Node, err error) {
	fComp, err := forwarderInitFunc()
	if err != nil {
		return
	} else if fComp == nil {
		err = fmt.Errorf("expected forwarderInitFunc not to return nil")
		return
	}

	return &Node{
		forwarder: fComp,
	}, nil
}

func (n *Node) Forwarder() qc.Forwarder {
	return n.forwarder
}

func (n *Node) Result() qt.Tensor {
	return n.result
}

func (n *Node) Parents() []*Node {
	return makeCopy(n.parents)
}

func (n *Node) Children() []*Node {
	return makeCopy(n.children)
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

	y, err := n.forwarder.Forward(xs...)
	if err != nil {
		return
	}

	n.result = y

	return nil
}

func (n *Node) Optimize(optimizer qc.Optimizer) (err error) {
	wfComp, ok := n.forwarder.(qc.WeightedForwarder)
	if !ok {
		return nil
	}

	for _, w := range wfComp.TrainableWeights() {
		err = optimizer.Update(w)
		if err != nil {
			return
		}
	}

	return nil
}

/* ----- helpers ----- */

func makeCopy[T any](src []T) (dst []T) {
	dst = make([]T, len(src))
	copy(dst, src)
	return dst
}
