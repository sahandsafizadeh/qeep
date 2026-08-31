package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor/internal/core"

type gctxTensor interface {
	core.Tensor
	GradientContext() *GradContext
}

type GradContext struct {
	tracked   bool
	bpdirty   bool
	gradient  core.Tensor
	backEdges []*backwardEdge
}

type backwardEdge struct {
	target core.Tensor
	gradFn chainGradFunc
}

type backpropState struct {
	unconsumed int
	grsnapshot core.Tensor
}

type chainGradFunc func() (core.Tensor, error)
