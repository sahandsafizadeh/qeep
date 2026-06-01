package metrics

import (
	"fmt"
	"math"

	"github.com/sahandsafizadeh/qeep/tensor"
)

// MSE tracks squared errors over accumulated batches.
type MSE struct {
	count   int
	diffSum float64
}

func NewMSE() *MSE {
	return new(MSE)
}

func (c *MSE) Accumulate(yp tensor.Tensor, yt tensor.Tensor) (err error) {
	err = c.validateInputs(yp, yt)
	if err != nil {
		return fmt.Errorf("MSE input data validation failed: %w", err)
	}

	err = c.accumulate(yp, yt)
	if err != nil {
		return fmt.Errorf("MSE accumulate failed: %w", err)
	}

	return nil
}

func (c *MSE) accumulate(yp tensor.Tensor, yt tensor.Tensor) (err error) {
	diff, err := yt.Sub(yp)
	if err != nil {
		return err
	}

	diff = diff.Pow(2)

	shape := diff.Shape()

	c.count += shape[0] * shape[1]
	c.diffSum += diff.Sum()

	return nil
}

func (c *MSE) Result() float64 {
	if c.count == 0 {
		return math.NaN()
	}

	return c.diffSum / float64(c.count)
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
