package model

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/internal/queue"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
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
	q.Enqueue(roots...)

	for !q.IsEmpty() {
		cn, err := q.Dequeue()
		if err != nil {
			panic(fmt.Sprintf("traverseBFS: dequeue failed on non-empty queue: %v", err))
		}

		err = applyFunc(cn)
		if err != nil {
			return fmt.Errorf("(Layer %d): %w", cn.NLayer(), err)
		}

		q.Enqueue(cn.Children()...)
	}

	return nil
}
