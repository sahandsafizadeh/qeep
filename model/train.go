package model

import (
	"fmt"

	qc "github.com/sahandsafizadeh/qeep/component"
	"github.com/sahandsafizadeh/qeep/model/internal/queue"
	"github.com/sahandsafizadeh/qeep/model/node"
	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

type Model struct {
	roots     []node.Node
	leaf      node.Node
	lossFunc  qc.LossFunc
	optimFunc qc.OptimizerFunc
}

func (m *Model) Fit(x qt.Tensor, y qt.Tensor) (err error) {
	for i := range 300 {
		// batch generation

		err = trainStep(m, x, y)
		if err != nil {
			return
		}

		fmt.Println("Epoch:", i)
	}

	return nil
}

func (m *Model) Predict(x qt.Tensor) (o qt.Tensor, err error) {
	// create input node
	_ = x

	err = m.forward()
	if err != nil {
		return
	}

	return m.leaf.Output(), nil
}

func trainStep(m *Model, x qt.Tensor, y qt.Tensor) (err error) {
	// create input node
	_ = x

	err = m.forward()
	if err != nil {
		return
	}

	l, err := m.lossFunc(m.leaf.Output(), y)
	if err != nil {
		return
	}

	err = tinit.BackProp(l)
	if err != nil {
		return
	}

	err = m.optimize()
	if err != nil {
		return
	}

	return nil
}

func (m *Model) forward() (err error) {
	return traverseBFS(m.roots, func(n node.Node) error { return n.Forward() })
}

func (m *Model) optimize() (err error) {
	return traverseBFS(m.roots, func(n node.Node) error { return n.Optimize(m.optimFunc) })
}

func traverseBFS(roots []node.Node, applyFunc func(node.Node) error) (err error) {
	q := queue.NewQueue[node.Node]()
	q.Enqueue(roots)

	var cn node.Node

	for !q.IsEmpty() {
		cn, err = q.Dequeue()
		if err != nil {
			return
		}

		err = applyFunc(cn)
		if err != nil {
			return
		}

		q.Enqueue(cn.Children())
	}

	return nil
}
