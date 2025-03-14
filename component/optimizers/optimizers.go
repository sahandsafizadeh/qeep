package optimizers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func getValidOptimizerInputs(wptr *tensor.Tensor) (w tensor.Tensor, g tensor.Tensor, err error) {
	w = *wptr
	g = w.Gradient()

	if g == nil {
		err = fmt.Errorf("expected tensor's gradient not to be nil")
		return
	}

	return w, g, nil
}
