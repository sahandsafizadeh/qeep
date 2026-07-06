package model

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/internal/queue"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
)

func (m *Model) forward() (err error) {
	return traverse(m.inputs, func(n *node.Node) error {
		return n.Forward()
	})
}

func (m *Model) optimize() (err error) {
	return traverse(m.inputs, func(n *node.Node) error {
		return n.Optimize(m.optimizer)
	})
}

func (m *Model) disableGrad() (err error) {
	return traverse(m.inputs, func(n *node.Node) error {
		n.DisableGrad()
		return nil
	})
}

func (m *Model) enableGrad() (err error) {
	return traverse(m.inputs, func(n *node.Node) error {
		n.EnableGrad()
		return nil
	})
}

func traverse(roots []*node.Node, applyFunc func(*node.Node) error) (err error) {
	unconsumed := prepareTraverseStates(roots)

	q := queue.NewQueue[*node.Node]()
	q.Enqueue(roots...)

	for !q.IsEmpty() {
		pn := q.Dequeue()

		err = applyFunc(pn)
		if err != nil {
			return fmt.Errorf("(Layer %d): %w", pn.NLayer(), err)
		}

		for _, cn := range pn.Children() {
			unconsumed[cn]--
			if unconsumed[cn] == 0 {
				q.Enqueue(cn)
			}
		}
	}

	return nil
}

func prepareTraverseStates(roots []*node.Node) map[*node.Node]int {
	states := make(map[*node.Node]int)

	q := queue.NewQueue[*node.Node]()
	q.Enqueue(roots...)

	for !q.IsEmpty() {
		pn := q.Dequeue()
		for _, cn := range pn.Children() {
			if _, seen := states[cn]; !seen {
				states[cn] = len(cn.Parents())
				q.Enqueue(cn)
			}
		}
	}

	return states
}
