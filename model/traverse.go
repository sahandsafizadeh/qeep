package model

import (
	"github.com/sahandsafizadeh/qeep/model/internal/node"
	"github.com/sahandsafizadeh/qeep/model/internal/queue"
)

func (m *Model) forward() (err error) {
	return traverseBFS(m.inputs, func(n *node.Node) error {
		return n.Forward()
	})
}

func (m *Model) optimize() (err error) {
	return traverseBFS(m.inputs, func(n *node.Node) error {
		return n.Optimize(m.optimizer)
	})
}

func (m *Model) disableGrad() (err error) {
	return traverseBFS(m.inputs, func(n *node.Node) error {
		n.DisableGrad()
		return nil
	})
}

func (m *Model) enableGrad() (err error) {
	return traverseBFS(m.inputs, func(n *node.Node) error {
		n.EnableGrad()
		return nil
	})
}

func traverseBFS(roots []*node.Node, applyFunc func(*node.Node) error) (err error) {
	q := queue.NewQueue[*node.Node]()
	q.Enqueue(roots)

	var cn *node.Node

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
