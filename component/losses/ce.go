package losses

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

const epsilon = 1e-12

type CE struct {
}

func NewCE() (c *CE) {
	return new(CE)
}

func (c *CE) Compute(yp tensor.Tensor, yt tensor.Tensor) (l tensor.Tensor, err error) {
	err = c.validateInputs(yp, yt)
	if err != nil {
		err = fmt.Errorf("CE input data validation failed: %w", err)
		return
	}

	/* ----- clip tensors ----- */

	yt, err = clip(yt, 0, 1)
	if err != nil {
		return
	}

	yp, err = clip(yp, epsilon, 1-epsilon)
	if err != nil {
		return
	}

	/* ----- sentence ----- */

	s, err := yt.Mul(yp.Log())
	if err != nil {
		return
	}

	/* ----- aggregation ----- */

	l, err = s.SumAlong(1)
	if err != nil {
		return
	}

	l = l.Scale(-1)

	return l.MeanAlong(0)
}

func clip(x tensor.Tensor, l, u float64) (y tensor.Tensor, err error) {
	_1 := x.Pow(0)
	lower := _1.Scale(l)
	upper := _1.Scale(u)

	y, err = x.ElMin(upper)
	if err != nil {
		return
	}

	y, err = lower.ElMax(y)
	if err != nil {
		return
	}

	return y, nil
}

/* ----- helpers ----- */

func (c *CE) validateInputs(yp tensor.Tensor, yt tensor.Tensor) (err error) {
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

	if shapep[1] != shapet[1] {
		err = fmt.Errorf("expected input tensor sizes to match along class dimension: (%d) != (%d)", shapep[1], shapet[1])
		return
	}

	return nil
}
