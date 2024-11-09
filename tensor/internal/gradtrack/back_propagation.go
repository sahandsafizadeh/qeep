package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor"

func BackPropagate(t tensor.Tensor) (err error) {
	return backward(startEdge(t))
}

func startEdge(t tensor.Tensor) (edge *backwardEdge) {
	return &backwardEdge{
		target: t,
		gradFn: func() (tensor.Tensor, error) {
			// neutral tensor; same shape, all ones
			return toOnes(t), nil
		},
	}
}

func backward(edge *backwardEdge) (err error) {
	gctx := gradContextOf(edge.target)

	if !gctx.tracked {
		return nil
	} else {
		gctx.bpdirty = true
	}

	grad, err := edge.gradFn()
	if err != nil {
		return
	}

	err = accumulateGrad(gctx, grad)
	if err != nil {
		return
	}

	for _, e := range gctx.backEdges {
		err = backward(e)
		if err != nil {
			return
		}
	}

	return nil
}

func accumulateGrad(gctx *GradContext, grad tensor.Tensor) (err error) {
	if gctx.gradient == nil {
		gctx.gradient = grad
	} else {
		gctx.gradient, err = gctx.gradient.Add(grad)
		if err != nil {
			return
		}
	}

	return nil
}
