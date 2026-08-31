package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor/internal/core"

type gctxTensor interface {
	core.Tensor
	GradientContext() *GradContext
}

type GradContext struct {
	tracked   bool
	bpdirty   bool
	gradient  gctxTensor
	backEdges []*backwardEdge
}

type backwardEdge struct {
	target gctxTensor
	gradFn chainGradFunc
}

type backpropState struct {
	unconsumed int
	grsnapshot gctxTensor
}

type chainGradFunc func() (gctxTensor, error)
