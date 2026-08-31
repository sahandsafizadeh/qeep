package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor/internal/core"

func NewGradContext(tracked bool) *GradContext {
	return &GradContext{tracked: tracked}
}

func NewDirtyGradContext() *GradContext {
	gctx := NewGradContext(false)
	gctx.bpdirty = true
	return gctx
}

func (gctx *GradContext) Tracked() bool {
	return gctx.tracked
}

func (gctx *GradContext) Gradient() core.Tensor {
	return gctx.gradient
}

/* ----- helpers ----- */

func anyIsBPDirty(ts ...GctxEnabledTensor) bool {
	for _, t := range ts {
		gctx := t.GradientContext()
		if gctx.bpdirty {
			return true
		}
	}

	return false
}

func nonIsTracked(ts ...GctxEnabledTensor) bool {
	for _, t := range ts {
		gctx := t.GradientContext()
		if gctx.tracked {
			return false
		}
	}

	return true
}
