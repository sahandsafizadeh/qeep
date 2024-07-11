package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor"

type GradContext struct {
	grad           tensor.Tensor
	backEdges      []*backwardEdge
	trackForbidden bool
}

type backwardEdge struct {
	target tensor.Tensor
	gradFn chainGradFunc
}

type chainGradFunc func() (tensor.Tensor, error)
