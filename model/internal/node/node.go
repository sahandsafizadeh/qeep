package node

import (
	"github.com/sahandsafizadeh/qeep/model/internal/contract"
	"github.com/sahandsafizadeh/qeep/tensor"
)

type Node struct {
	layer    contract.Layer
	result   tensor.Tensor
	parents  []*Node
	children []*Node
}

func NewNode(layer contract.Layer) (n *Node) {
	return &Node{layer: layer}
}

func (n *Node) Layer() contract.Layer {
	return n.layer
}

func (n *Node) Result() tensor.Tensor {
	return n.result
}

func (n *Node) Parents() []*Node {
	return n.parents
}

func (n *Node) Children() []*Node {
	return n.children
}

func (n *Node) AddParent(p *Node) {
	n.parents = append(n.parents, p)
}

func (n *Node) AddChild(c *Node) {
	n.children = append(n.children, c)
}

func (n *Node) Forward() (err error) {
	xs := make([]tensor.Tensor, len(n.parents))
	for i, p := range n.parents {
		xs[i] = p.result
	}

	y, err := n.layer.Forward(xs...)
	if err != nil {
		return
	}

	n.result = y

	return nil
}

func (n *Node) Optimize(optimizer contract.Optimizer) (err error) {
	wl, ok := n.layer.(contract.WeightedLayer)
	if !ok {
		return nil
	}

	for _, w := range wl.Weights() {
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

func (n *Node) DisableGrad() {
	wl, ok := n.layer.(contract.WeightedLayer)
	if !ok {
		return
	}

	for _, w := range wl.Weights() {
		(*w.Value).ResetGradContext(false)
	}
}

func (n *Node) EnableGrad() {
	wl, ok := n.layer.(contract.WeightedLayer)
	if !ok {
		return
	}

	for _, w := range wl.Weights() {
		(*w.Value).ResetGradContext(w.Trainable)
	}
}
