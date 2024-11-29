package losses

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type BCE struct {
}

func NewBCE() (c *BCE) {
	return new(BCE)
}

func (c *BCE) Compute(yp tensor.Tensor, yt tensor.Tensor) (l tensor.Tensor, err error) {
	err = c.validateInputs(yp, yt)
	if err != nil {
		err = fmt.Errorf("BCE input data validation failed: %w", err)
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

	/* ----- left sentence ----- */

	t1 := yt
	y1 := yp

	s1, err := t1.Mul(y1.Log())
	if err != nil {
		return
	}

	/* ----- right sentence ----- */

	_1 := yp.Pow(0)

	t2, err := _1.Sub(yt)
	if err != nil {
		return
	}

	y2, err := _1.Sub(yp)
	if err != nil {
		return
	}

	s2, err := t2.Mul(y2.Log())
	if err != nil {
		return
	}

	/* ----- aggregation ----- */

	l, err = s1.Add(s2)
	if err != nil {
		return
	}

	l = l.Scale(-1.)

	return l.MeanAlong(0)
}

/* ----- helpers ----- */

func (c *BCE) validateInputs(yp tensor.Tensor, yt tensor.Tensor) (err error) {
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