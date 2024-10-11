package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor"

func (gctx *GradContext) Grad() (g tensor.Tensor) {
	if !isTrackRequired(gctx) {
		return nil
	}

	return gctx.grad
}

func ForbiddenForAny(ts ...tensor.Tensor) (ok bool) {
	for _, t := range ts {
		gctx := gradContextOf(t)
		if isTrackForbidden(gctx) {
			return true
		}
	}

	return false
}

func RequiredForAny(ts ...tensor.Tensor) (ok bool) {
	for _, t := range ts {
		gctx := gradContextOf(t)
		if isTrackRequired(gctx) {
			return true
		}
	}

	return false
}

func gradContextOf(t tensor.Tensor) (gctx *GradContext) {
	return t.GradContext().(*GradContext)
}

func isTrackForbidden(gctx *GradContext) (ok bool) {
	if gctx == nil {
		return false
	}

	return gctx.trackForbidden
}

func isTrackRequired(gctx *GradContext) (ok bool) {
	return gctx != nil
}
