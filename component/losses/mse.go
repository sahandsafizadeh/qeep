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

	d, err := yt.Sub(yp)
	if err != nil {
		return
	}

	d = d.Pow(2)

	return d.MeanAlong(0)
}

/* ----- helpers ----- */

func (c *MSE) validateInputs(yp tensor.Tensor, yt tensor.Tensor) (err error) {
	if yp == nil || yt == nil {
		err = fmt.Errorf("expected input tensors not to be nil")
		return
	}

	shapep := yp.Shape()
	shapet := yt.Shape()

	if len(shapep) != 1 || len(shapet) != 1 {
		err = fmt.Errorf("expected input tensors to have exactly one dimension (batch)")
		return
	}

	if shapep[0] != shapet[0] {
		err = fmt.Errorf("expected input tensor sizes to match along batch dimension: (%d) != (%d)", shapep[0], shapet[0])
		return
	}

	return nil
}
