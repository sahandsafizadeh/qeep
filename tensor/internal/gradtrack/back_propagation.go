package gradtrack

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/internal/queue"
	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
)

func BackPropagate(t tensor.Tensor) (err error) {
	err = backpropagate(t)
	if err != nil {
		return fmt.Errorf("BackPropagate: %w", err)
	}

	return nil
}

func backpropagate(t tensor.Tensor) (err error) {
	states := prepareBackpropStates(t)

	err = backpropRTS(t, states)
	if err != nil {
		return err
	}

	err = accumulateGradSnapshots(states)
	if err != nil {
		return err
	}

	return nil
}

func prepareBackpropStates(root tensor.Tensor) map[*GradContext]*backpropState {
	panic("unimplemented")
}

func backpropRTS(root tensor.Tensor, states map[*GradContext]*backpropState) (err error) {
	q := queue.NewQueue[*backwardEdge]()
	q.Enqueue(&backwardEdge{
		target: root,
		gradFn: func() (tensor.Tensor, error) {
			return toOnes(root), nil // neutral tensor; same shape, all ones
		},
	})

	for !q.IsEmpty() {
		src, err := q.Dequeue()
		if err != nil {
			panic(fmt.Sprintf("prepareBackpropStates: dequeue failed on non-empty queue: %v", err))
		}

		for _, edge := range src.backEdges {
			dst := gradContextOf(edge.target)
			if !dst.tracked {
				continue
			}

			bpst, seen := states[dst]
			if !seen {
				bpst = &backpropState{
					unconsumed: 0,
					grsnapshot: dst.gradient,
				}

				states[dst] = bpst
				dst.gradient = nil
				q.Enqueue(dst)
			}

			bpst.unconsumed++
		}
	}

	return states
}

func accumulateGradSnapshots(states map[*GradContext]*backpropState) (err error) {
	for gctx, bpst := range states {
		if bpst.grsnapshot != nil {
			err = accumulateGrad(gctx, bpst.grsnapshot)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

/* ----- helpers ----- */

func accumulateGrad(gctx *GradContext, grad tensor.Tensor) (err error) {
	if gctx.gradient == nil {
		gctx.gradient = grad
	} else {
		acc, err := gctx.gradient.Add(grad)
		if err != nil {
			return err
		}

		gctx.gradient = acc
	}

	return nil
}
