package model

import (
	"fmt"

	qc "github.com/sahandsafizadeh/qeep/component"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
	"github.com/sahandsafizadeh/qeep/model/internal/queue"
	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func NewModel() (*Model, error) {
	// garuntee that model at least has one input
	// garuntee that every input node is of type input component
	// garuntee that every node in model is not null and initialized properly
	return nil, nil
}

// requires batch dimension in all inputs
func (m *Model) Predict(xs []qt.Tensor) (yp qt.Tensor, err error) {

	// TODO: validate batch consistency of xs

	err = m.seed(xs)
	if err != nil {
		return
	}

	err = m.forward()
	if err != nil {
		return
	}

	return m.output.Result(), nil
}

func (m *Model) Fit(xs []qt.Tensor, yt qt.Tensor, conf *FitConfig) (err error) {
	for range conf.Epochs {
		nr := xs[0].Shape()[0]
		nb := (conf.BatchSize / nr) + 1

		for i := range nb {
			xbs := make([]qt.Tensor, 0, conf.BatchSize)
			for _, x := range xs {
				var y qt.Tensor

				y, err = x.Slice([]qt.Range{{From: i * conf.BatchSize, To: (i + 1) * conf.BatchSize}})
				if err != nil {
					return
				}

				xbs = append(xbs, y)
			}

			err = m.fitStep(xbs, yt)
			if err != nil {
				return
			}
		}
	}

	return nil
}

func (m *Model) seed(xs []qt.Tensor) (err error) {
	if len(xs) != len(m.inputs) {
		err = fmt.Errorf("expected same number of inputs")
		return
	}

	for i, n := range m.inputs {
		c := n.Component()
		c1 := c.(*qc.Input)
		c1.SeedFunc = func() qt.Tensor {
			return xs[i]
		}
	}

	return nil
}

func (m *Model) fitStep(xs []qt.Tensor, yt qt.Tensor) (err error) {
	yp, err := m.Predict(xs)
	if err != nil {
		return
	}

	l, err := m.lossFunc(yp, yt)
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
	return traverseBFS(m.inputs, func(n *node.Node) error { return n.Forward() })
}

func (m *Model) optimize() (err error) {
	return traverseBFS(m.inputs, func(n *node.Node) error { return n.Optimize(m.optimFunc) })
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
