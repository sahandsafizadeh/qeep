package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor/internal/tensor"

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

func (gctx *GradContext) Gradient() tensor.Tensor {
	return gctx.gradient
}

/* ----- helpers ----- */

func anyIsBPDirty(ts ...tensor.Tensor) bool {
	for _, t := range ts {
		gctx := gradContextOf(t)
		if gctx.bpdirty {
			return true
		}
	}

	return false
}

func nonIsTracked(ts ...tensor.Tensor) bool {
	for _, t := range ts {
		gctx := gradContextOf(t)
		if gctx.tracked {
			return false
		}
	}

	return true
}

func gradContextOf(t tensor.Tensor) *GradContext {
	return t.GradContext().(*GradContext)
}
