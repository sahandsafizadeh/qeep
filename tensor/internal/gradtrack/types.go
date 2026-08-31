package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor/internal/core"

type GctxEnabledTensor interface {
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
	target GctxEnabledTensor
	gradFn chainGradFunc
}

type backpropState struct {
	unconsumed int
	grsnapshot core.Tensor
}

type chainGradFunc func() (core.Tensor, error)
