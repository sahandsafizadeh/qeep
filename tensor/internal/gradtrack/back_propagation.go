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
			// neutral tensor; same shape, all ones
			return toOnes(t), nil
		},
	}
}

func backward(edge *backwardEdge) error {
	gctx := gradContextOf(edge.target)

	if !gctx.tracked {
		return nil
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

	for _, e := range gctx.backEdges {
		err = backward(e)
		if err != nil {
			return err
		}
	}

	return nil
}

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
