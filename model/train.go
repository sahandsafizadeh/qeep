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

type nodeFunc func(node.Node) error

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

func (m *Model) Predict() (o qt.Tensor, err error) {
	return
}

func trainStep(m *Model, x qt.Tensor, y qt.Tensor) (err error) {
	// create input node
	_ = x

	err = applyBFS(m.roots, func(n node.Node) error { return n.Forward() })
	if err != nil {
		return
	}

	yt := y
	yp := m.leaf.Output()

	l, err := m.lossFunc(yp, yt)
	if err != nil {
		return
	}

	err = tinit.BackProp(l)
	if err != nil {
		return
	}

	err = applyBFS(m.roots, func(n node.Node) error { return n.Optimize(m.optimFunc) })
	if err != nil {
		return
	}

	return nil
}

func applyBFS(roots []node.Node, applyFn nodeFunc) (err error) {
	q := queue.NewQueue[node.Node]()
	q.Enqueue(roots)

	var cn node.Node

	for !q.IsEmpty() {
		cn, err = q.Dequeue()
		if err != nil {
			return
		}

		err = applyFn(cn)
		if err != nil {
			return
		}

		q.Enqueue(cn.Children())
	}

	return nil
}
