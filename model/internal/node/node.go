package node

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/model/internal/types"
	"github.com/sahandsafizadeh/qeep/tensor"
)

type Node struct {
	forwarder types.Forwarder
	result    tensor.Tensor
	parents   []*Node
	children  []*Node
}

func NewNode(forwarder types.Forwarder) (n *Node, err error) {
	return &Node{forwarder: forwarder}, nil
}

func (n *Node) Forwarder() types.Forwarder {
	return n.forwarder
}

func (n *Node) Result() tensor.Tensor {
	return n.result
}

func (n *Node) Parents() []*Node {
	return makeCopy(n.parents)
}

func (n *Node) Children() []*Node {
	return makeCopy(n.children)
}

func (n *Node) AddParent(p *Node) (err error) {
	err = validateInputNode(p)
	if err != nil {
		return
	}

	n.parents = append(n.parents, p)

	return nil
}

func (n *Node) AddChild(c *Node) (err error) {
	err = validateInputNode(c)
	if err != nil {
		return
	}

	n.children = append(n.children, c)

	return nil
}

func (n *Node) Forward() (err error) {
	xs := make([]tensor.Tensor, len(n.parents))
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

func (n *Node) Optimize(optimizer types.Optimizer) (err error) {
	wf, ok := n.forwarder.(types.WeightedForwarder)
	if !ok {
		return nil
	}

	for _, w := range wf.Weights() {
		if !w.Trainable {
			continue
		}

		err = optimizer.Update(w.Value)
		if err != nil {
			return
		}
	}

	return nil
}

func (n *Node) DisableGrad() (err error) {
	wf, ok := n.forwarder.(types.WeightedForwarder)
	if !ok {
		return nil
	}

	for _, w := range wf.Weights() {
		(*w.Value).ResetGradContext(false)
	}

	return nil
}

func (n *Node) EnableGrad() (err error) {
	wf, ok := n.forwarder.(types.WeightedForwarder)
	if !ok {
		return nil
	}

	for _, w := range wf.Weights() {
		(*w.Value).ResetGradContext(w.Trainable)
	}

	return nil
}

/* ----- helpers ----- */

func validateInputNode(n *Node) (err error) {
	if n == nil {
		err = fmt.Errorf("expected input node not to be nil")
		return
	}

	return nil
}

func makeCopy[T any](src []T) (dst []T) {
	dst = make([]T, len(src))
	copy(dst, src)
	return dst
}
