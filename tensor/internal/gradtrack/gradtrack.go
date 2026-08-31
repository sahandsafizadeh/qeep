package gradtrack

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

func (gctx *GradContext) Gradient() gctxTensor {
	return gctx.gradient
}

/* ----- helpers ----- */

func anyIsBPDirty(ts ...gctxTensor) bool {
	for _, t := range ts {
		gctx := t.GradientContext()
		if gctx.bpdirty {
			return true
		}
	}

	return false
}

func nonIsTracked(ts ...gctxTensor) bool {
	for _, t := range ts {
		gctx := t.GradientContext()
		if gctx.tracked {
			return false
		}
	}

	return true
}
