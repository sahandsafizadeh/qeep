package losses

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

const epsilon = 1e-12

// CE is cross-entropy loss for multi-class classification.
type CE struct {
}

func NewCE() *CE {
	return new(CE)
}

func (c *CE) Compute(yp tensor.Tensor, yt tensor.Tensor) (l tensor.Tensor, err error) {
	err = c.validateInputs(yp, yt)
	if err != nil {
		return l, fmt.Errorf("CE input data validation failed: %w", err)
	}

	l, err = c.compute(yp, yt)
	if err != nil {
		return l, fmt.Errorf("CE compute failed: %w", err)
	}

	return l, nil
}

func (c *CE) compute(yp tensor.Tensor, yt tensor.Tensor) (l tensor.Tensor, err error) {
	/* ----- clip tensors ----- */

	yt, err = clip(yt, 0, 1)
	if err != nil {
		return l, err
	}

	yp, err = clip(yp, epsilon, 1-epsilon)
	if err != nil {
		return l, err
	}

	/* ----- sentence ----- */

	s, err := yt.Mul(yp.Log())
	if err != nil {
		return l, err
	}

	/* ----- aggregation ----- */

	l, err = s.SumAlong(1)
	if err != nil {
		return l, err
	}

	l = l.Scale(-1)

	return l.MeanAlong(0)
}

func clip(x tensor.Tensor, l, u float64) (y tensor.Tensor, err error) {
	dev := x.Device()
	dims := x.Shape()

	_1, err := tensor.Ones(dims, &tensor.Config{
		Device:    dev,
		GradTrack: false,
	})
	if err != nil {
		return y, err
	}

	lower := _1.Scale(l)
	upper := _1.Scale(u)

	y, err = x.ElMin(upper)
	if err != nil {
		return y, err
	}

	y, err = lower.ElMax(y)
	if err != nil {
		return y, err
	}

	return y, nil
}

/* ----- helpers ----- */

func (c *CE) validateInputs(yp tensor.Tensor, yt tensor.Tensor) (err error) {
	shapep := yp.Shape()
	shapet := yt.Shape()

	if len(shapep) != 2 || len(shapet) != 2 {
		return fmt.Errorf("expected input tensors to have exactly two dimensions (batch, class)")
	}

	if shapep[0] != shapet[0] {
		return fmt.Errorf("expected input tensor sizes to match along batch dimension: (%d) != (%d)", shapep[0], shapet[0])
	}

	if shapep[1] != shapet[1] {
		return fmt.Errorf("expected input tensor sizes to match along class dimension: (%d) != (%d)", shapep[1], shapet[1])
	}

	return nil
}
