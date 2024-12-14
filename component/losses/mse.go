package losses

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type MSE struct {
}

func NewMSE() (c *MSE) {
	return new(MSE)
}

func (c *MSE) Compute(yp tensor.Tensor, yt tensor.Tensor) (l tensor.Tensor, err error) {
	err = c.validateInputs(yp, yt)
	if err != nil {
		err = fmt.Errorf("MSE input data validation failed: %w", err)
		return
	}

	l, err = yt.Sub(yp)
	if err != nil {
		return
	}

	l, err = l.Squeeze(1)
	if err != nil {
		return
	}

	l = l.Pow(2)

	return l.MeanAlong(0)
}

/* ----- helpers ----- */

func (c *MSE) validateInputs(yp tensor.Tensor, yt tensor.Tensor) (err error) {
	if yp == nil || yt == nil {
		err = fmt.Errorf("expected input tensors not to be nil")
		return
	}

	shapep := yp.Shape()
	shapet := yt.Shape()

	if len(shapep) != 2 || len(shapet) != 2 {
		err = fmt.Errorf("expected input tensors to have exactly two dimensions (batch, class)")
		return
	}

	if shapep[0] != shapet[0] {
		err = fmt.Errorf("expected input tensor sizes to match along batch dimension: (%d) != (%d)", shapep[0], shapet[0])
		return
	}

	if shapep[1] != 1 || shapet[1] != 1 {
		err = fmt.Errorf("expected input tensor sizes to be equal to (1) along class dimension")
		return
	}

	return nil
}
