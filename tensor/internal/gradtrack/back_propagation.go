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
	root := createBackpropRoot(t)
	states := prepareBackpropStates(root)

	err = backpropRTS(root, states)
	if err != nil {
		return err
	}

	err = accumulateGradSnapshots(states)
	if err != nil {
		return err
	}

	return nil
}

func createBackpropRoot(t tensor.Tensor) *GradContext {
	return &GradContext{
		tracked: false,
		backEdges: []*backwardEdge{
			{
				target: t,
				gradFn: func() (tensor.Tensor, error) {
					// neutral tensor; same shape, all ones
					return toOnes(t), nil
				},
			},
		},
	}
}

func prepareBackpropStates(root *GradContext) map[*GradContext]*backpropState {
	states := make(map[*GradContext]*backpropState)

	q := queue.NewQueue[*GradContext]()
	q.Enqueue(root)

	for !q.IsEmpty() {
		src := q.Dequeue()
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

func backpropRTS(root *GradContext, states map[*GradContext]*backpropState) (err error) {
	q := queue.NewQueue[*GradContext]()
	q.Enqueue(root)

	for !q.IsEmpty() {
		src := q.Dequeue()
		for _, edge := range src.backEdges {
			dst := gradContextOf(edge.target)
			if !dst.tracked {
				continue
			}

			// crucial for preventing memory leak
			dst.bpdirty = true

			grad, err := edge.gradFn()
			if err != nil {
				return err
			}

			err = accumulateGrad(dst, grad)
			if err != nil {
				return err
			}

			bpst := states[dst]
			bpst.unconsumed--
			if bpst.unconsumed == 0 {
				q.Enqueue(dst)
			}
		}
	}

	return nil
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

func accumulateGrad(gctx *GradContext, grad tensor.Tensor) (err error) {
	ensureGradSafeToAccumulate(grad)

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

func ensureGradSafeToAccumulate(grad tensor.Tensor) {
	gctx := gradContextOf(grad)
	if !gctx.bpdirty || len(gctx.backEdges) > 0 {
		panic("gradtrack: memory leak danger: gradient tensor must be dirty and untracked")
	}
}
