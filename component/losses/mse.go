package losses

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

// MSE is mean squared error loss. Used for regression.
type MSE struct {
}

func NewMSE() *MSE {
	return new(MSE)
}

func (c *MSE) Compute(yp tensor.Tensor, yt tensor.Tensor) (l tensor.Tensor, err error) {
	err = c.validateInputs(yp, yt)
	if err != nil {
		return l, fmt.Errorf("MSE input data validation failed: %w", err)
	}

	l, err = c.compute(yp, yt)
	if err != nil {
		return l, fmt.Errorf("MSE compute failed: %w", err)
	}

	return l, nil
}

func (c *MSE) compute(yp tensor.Tensor, yt tensor.Tensor) (l tensor.Tensor, err error) {
	l, err = yt.Sub(yp)
	if err != nil {
		return l, err
	}

	l = l.Pow(2)

	l, err = l.MeanAlong(1)
	if err != nil {
		return l, err
	}

	return l.MeanAlong(0)
}

/* ----- helpers ----- */

func (c *MSE) validateInputs(yp tensor.Tensor, yt tensor.Tensor) (err error) {
	shapep := yp.Shape()
	shapet := yt.Shape()

	if len(shapep) != 2 || len(shapet) != 2 {
		return fmt.Errorf("expected input tensors to have exactly two dimensions (batch, data)")
	}

	if shapep[0] != shapet[0] {
		return fmt.Errorf("expected input tensor sizes to match along batch dimension: (%d) != (%d)", shapep[0], shapet[0])
	}

	if shapep[1] != shapet[1] {
		return fmt.Errorf("expected input tensor sizes to match along data dimension: (%d) != (%d)", shapep[1], shapet[1])
	}

	return nil
}
