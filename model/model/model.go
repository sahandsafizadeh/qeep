package model

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/model/input"
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
	// validate config
	// validate sizes of xs and yt match

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

			err = m.fitStepToBeRenamed(xbs, yt)
			if err != nil {
				return
			}
		}
	}

	return nil
}

/* ----- helpers ----- */

func (m *Model) seed(xs []qt.Tensor) (err error) {
	if len(xs) != len(m.inputs) {
		err = fmt.Errorf("expected inputs length to match that of model: (%d) != (%d)", len(xs), len(m.inputs))
		return
	}

	batchDim := func(x qt.Tensor) int { return x.Shape()[0] }
	batchSize := batchDim(xs[0])

	for i, x := range xs {
		xbs := batchDim(x)
		if xbs != batchSize {
			err = fmt.Errorf("expected input tensors to have the same batch size: (%d) != (%d) for tensor (%d)", xbs, batchSize, i)
			return
		}
	}

	for i, n := range m.inputs {
		inFor := n.Forwarder().(*input.Input)
		inFor.SeedFunc = func() qt.Tensor { return xs[i] }
	}

	return nil
}

func (m *Model) fitStepToBeRenamed(xs []qt.Tensor, yt qt.Tensor) (err error) {
	yp, err := m.Predict(xs)
	if err != nil {
		return
	}

	loss, err := m.lossFunc.Compute(yp, yt)
	if err != nil {
		return
	}

	err = tinit.BackProp(loss)
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
	return traverseBFS(m.inputs, func(n *node.Node) error { return n.Optimize(m.optimizer) })
}

func traverseBFS(roots []*node.Node, applyFunc nodeApplyFunc) (err error) {
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
