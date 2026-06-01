package losses

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

// BCE is binary cross-entropy loss. Use for binary classification.
type BCE struct {
}

func NewBCE() *BCE {
	return new(BCE)
}

func (c *BCE) Compute(yp tensor.Tensor, yt tensor.Tensor) (l tensor.Tensor, err error) {
	err = c.validateInputs(yp, yt)
	if err != nil {
		return l, fmt.Errorf("BCE input data validation failed: %w", err)
	}

	l, err = c.compute(yp, yt)
	if err != nil {
		return l, fmt.Errorf("BCE compute failed: %w", err)
	}

	return l, nil
}

func (c *BCE) compute(yp tensor.Tensor, yt tensor.Tensor) (l tensor.Tensor, err error) {
	/* ----- clip tensors ----- */

	yt, err = clip(yt, 0, 1)
	if err != nil {
		return l, err
	}

	yp, err = clip(yp, epsilon, 1-epsilon)
	if err != nil {
		return l, err
	}

	/* ----- left sentence ----- */

	t1 := yt
	y1 := yp

	s1, err := t1.Mul(y1.Log())
	if err != nil {
		return l, err
	}

	/* ----- right sentence ----- */

	_1, err := c.toUntrackedFull(yp, 1)
	if err != nil {
		return l, err
	}

	t2, err := _1.Sub(yt)
	if err != nil {
		return l, err
	}

	y2, err := _1.Sub(yp)
	if err != nil {
		return l, err
	}

	s2, err := t2.Mul(y2.Log())
	if err != nil {
		return l, err
	}

	/* ----- aggregation ----- */

	l, err = s1.Add(s2)
	if err != nil {
		return l, err
	}

	l = l.Scale(-1.)

	l, err = l.Squeeze(1)
	if err != nil {
		return l, err
	}

	return l.MeanAlong(0)
}

/* ----- helpers ----- */

func (c *BCE) validateInputs(yp tensor.Tensor, yt tensor.Tensor) (err error) {
	shapep := yp.Shape()
	shapet := yt.Shape()

	if len(shapep) != 2 || len(shapet) != 2 {
		return fmt.Errorf("expected input tensors to have exactly two dimensions (batch, class=1)")
	}

	if shapep[0] != shapet[0] {
		return fmt.Errorf("expected input tensor sizes to match along batch dimension: (%d) != (%d)", shapep[0], shapet[0])
	}

	if shapep[1] != 1 || shapet[1] != 1 {
		return fmt.Errorf("expected input tensor sizes to be equal to (1) along class dimension")
	}

	return nil
}

func (c *BCE) toUntrackedFull(x tensor.Tensor, value float64) (y tensor.Tensor, err error) {
	dev := x.Device()
	dims := x.Shape()

	return tensor.Full(dims, value, &tensor.Config{
		Device:    dev,
		GradTrack: false,
	})
}
