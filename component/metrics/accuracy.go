package metrics

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Accuracy struct {
	total   int
	correct int
}

func NewAccuracy() (c *Accuracy) {
	return new(Accuracy)
}

func (c *Accuracy) Accumulate(yp tensor.Tensor, yt tensor.Tensor) (err error) {
	err = c.validateInputs(yp, yt)
	if err != nil {
		err = fmt.Errorf("Accuracy input data validation failed: %w", err)
		return
	}

	eq, err := yp.Eq(yt)
	if err != nil {
		return
	}

	c.total += eq.Shape()[0]
	c.correct += int(eq.Sum())

	return nil
}

func (c *Accuracy) Result() (result float64, err error) {
	if c.total == 0 {
		return 0., nil
	}

	return float64(c.correct) / float64(c.total), nil
}

/* ----- helpers ----- */

func (c *Accuracy) validateInputs(yp tensor.Tensor, yt tensor.Tensor) (err error) {
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
