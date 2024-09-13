package node

import (
	qc "github.com/sahandsafizadeh/qeep/component"
	qt "github.com/sahandsafizadeh/qeep/tensor"
)

func (n *Node) AddChild(c *Node) {
	n.children = append(n.children, c)
}

func (n *Node) Children() (children []*Node) {
	return n.children
}

func (n *Node) Output() (o qt.Tensor) {
	return n.output
}

func (n *Node) Forward() (err error) {
	inputs := make([]qt.Tensor, len(n.parents))
	for i, p := range n.parents {
		inputs[i] = p.Output()
	}

	o, err := n.component.Forward(inputs...)
	if err != nil {
		return err
	}

	n.output = o

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
