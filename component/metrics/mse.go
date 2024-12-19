package metrics

import (
	"fmt"
	"math"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type MSE struct {
	count   int
	diffSum float64
}

func NewMSE() (c *MSE) {
	return new(MSE)
}

func (c *MSE) Accumulate(yp tensor.Tensor, yt tensor.Tensor) (err error) {
	err = c.validateInputs(yp, yt)
	if err != nil {
		err = fmt.Errorf("MSE input data validation failed: %w", err)
		return
	}

	diff, err := yt.Sub(yp)
	if err != nil {
		return
	}

	diff = diff.Pow(2)

	shape := diff.Shape()

	c.count += shape[0] * shape[1]
	c.diffSum += diff.Sum()

	return nil
}

func (c *MSE) Result() (result float64, err error) {
	if c.count == 0 {
		return math.NaN(), nil
	}

	return c.diffSum / float64(c.count), nil
}

/* ----- helpers ----- */

func (c *MSE) validateInputs(yp tensor.Tensor, yt tensor.Tensor) (err error) {
	shapep := yp.Shape()
	shapet := yt.Shape()

	if len(shapep) != 2 || len(shapet) != 2 {
		err = fmt.Errorf("expected input tensors to have exactly two dimensions (batch, data)")
		return
	}

	if shapep[0] != shapet[0] {
		err = fmt.Errorf("expected input tensor sizes to match along batch dimension: (%d) != (%d)", shapep[0], shapet[0])
		return
	}

	if shapep[1] != shapet[1] {
		err = fmt.Errorf("expected input tensor sizes to match along data dimension: (%d) != (%d)", shapep[1], shapet[1])
		return
	}

	return nil
}
