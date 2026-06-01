package optimizers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func getValidOptimizerInputs(wptr *tensor.Tensor) (w tensor.Tensor, g tensor.Tensor, err error) {
	w = *wptr

	if w == nil {
		err = fmt.Errorf("test error")
		return
	}

	g = w.Gradient()

	if g == nil {
		return w, g, fmt.Errorf("expected tensor's gradient not to be nil")
	}

	return w, g, nil
}
