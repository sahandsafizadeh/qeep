package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor/internal/tensor"

type GradContext struct {
	tracked   bool
	bpdirty   bool
	gradient  tensor.Tensor
	backEdges []*backwardEdge
}

type backwardEdge struct {
	target tensor.Tensor
	gradFn chainGradFunc
}

type backpropState struct {
	unconsumed int
	snapshotgr tensor.Tensor
}

type chainGradFunc func() (tensor.Tensor, error)
