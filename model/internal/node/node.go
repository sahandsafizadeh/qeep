package node

import (
	"fmt"

	qc "github.com/sahandsafizadeh/qeep/component"
	qcw "github.com/sahandsafizadeh/qeep/component/weighteds"
	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type Node struct {
	component     qc.Component
	result        qt.Tensor
	parents       []*Node
	children      []*Node
	isInitialized bool
}

type ComponentCreatorFunc func() (qc.Component, error)

func NewNode(compCreatorFunc ComponentCreatorFunc) (n *Node, err error) {
	comp, err := compCreatorFunc()
	if err != nil {
		return
	} else if comp == nil {
		err = fmt.Errorf("expected compCreatorFunc not to return nil")
		return
	}

	return &Node{
		component: comp,
	}, nil
}

func (n *Node) Component() qc.Component {
	return n.component
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
	err = n.InitializeOnce()
	if err != nil {
		return
	}

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

func (n *Node) InitializeOnce() (err error) {
	wComp, ok := n.component.(qcw.WeightedComponent)
	if !ok {
		return nil
	}

	if n.isInitialized {
		return nil
	}

	err = wComp.InitWeights()
	if err != nil {
		return
	}

	n.isInitialized = true

	return nil
}

func (n *Node) Optimize(optimFunc qc.OptimizerFunc) (err error) {
	wComp, ok := n.component.(qcw.WeightedComponent)
	if !ok {
		return nil
	}

	for _, w := range wComp.TrainableWeights() {
		err = optimFunc(w)
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
