package gradtrack

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
)

func BackPropagate(t tensor.Tensor) error {
	err := backward(startEdge(t))
	if err != nil {
		return fmt.Errorf("BackPropagate: %w", err)
	}

	return nil
}

func startEdge(t tensor.Tensor) *backwardEdge {
	return &backwardEdge{
		target: t,
		gradFn: func() (tensor.Tensor, error) {
			return toOnes(root), nil // neutral tensor; same shape, all ones
		},
	})

	for !q.IsEmpty() {
		edge, err := q.Dequeue()
		if err != nil {
			panic(fmt.Sprintf("backpropRTS: dequeue failed on non-empty queue: %v", err))
		}

		gctx := gradContextOf(edge.target)
		if !gctx.tracked {
			continue
		} else {
			gctx.bpdirty = true
		}

		grad, err := edge.gradFn()
		if err != nil {
			return err
		}

		err = accumulateGrad(gctx, grad)
		if err != nil {
			return err
		}

		bpst := states[gctx]
		bpst.unconsumed--
		if bpst.unconsumed == 0 {
			q.Enqueue(gctx.backEdges...)
		}
	}

	return nil
}

func accumulateGradSnapshots(states map[*GradContext]*backpropState) (err error) {
	for gctx, bpst := range states {
		if bpst.snapshotgr != nil {
			err := accumulateGrad(gctx, bpst.snapshotgr)
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
