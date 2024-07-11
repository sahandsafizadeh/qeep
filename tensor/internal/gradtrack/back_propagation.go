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
	if !isTrackRequired(gctx) {
		return nil
	}

	gctx.trackForbidden = true

	grad, err := edge.gradFn()
	if err != nil {
		return
	}

	err = accumGrad(gctx, grad)
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

func accumGrad(gctx *GradContext, grad tensor.Tensor) (err error) {
	if gctx.grad == nil {
		gctx.grad = grad
	} else {
		gctx.grad, err = gctx.grad.Add(grad)
		if err != nil {
			return
		}
	}

	return nil
}
