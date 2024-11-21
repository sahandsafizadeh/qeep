package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor/internal/tensor"

func NewGradContext(tracked bool) (gctx *GradContext) {
	return &GradContext{tracked: tracked}
}

func NewDirtyGradContext() (gctx *GradContext) {
	gctx = NewGradContext(false)
	gctx.bpdirty = true
	return gctx
}

func (gctx *GradContext) Gradient() (g tensor.Tensor) {
	return gctx.gradient
}

/* ----- helpers ----- */

func anyIsBPDirty(ts ...tensor.Tensor) (ok bool) {
	for _, t := range ts {
		gctx := gradContextOf(t)
		if gctx.bpdirty {
			return true
		}
	}

	return false
}

func nonIsTracked(ts ...tensor.Tensor) (ok bool) {
	for _, t := range ts {
		gctx := gradContextOf(t)
		if gctx.tracked {
			return false
		}
	}

	return true
}

func gradContextOf(t tensor.Tensor) (gctx *GradContext) {
	return t.GradContext().(*GradContext)
}
