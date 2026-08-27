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

func anyIsBPDirty(ts ...core.Tensor) bool {
	for _, t := range ts {
		gctx := gradContextOf(t)
		if gctx.bpdirty {
			return true
		}
	}

	return false
}

func nonIsTracked(ts ...core.Tensor) bool {
	for _, t := range ts {
		gctx := gradContextOf(t)
		if gctx.tracked {
			return false
		}
	}

	return true
}

func gradContextOf(t core.Tensor) *GradContext {
	return t.GradContext().(*GradContext)
}
