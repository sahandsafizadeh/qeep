package loss

import qt "github.com/sahandsafizadeh/qeep/tensor"

type CE struct {
}

func NewCE() (c *CE) {
	return new(CE)
}

func (c *CE) Compute(yp qt.Tensor, yt qt.Tensor) (l qt.Tensor, err error) {
	l, err = yt.Mul(yp.Log())
	if err != nil {
		return
	}

	l, err = l.SumAlong(1)
	if err != nil {
		return
	}

	l, err = l.MeanAlong(0)
	if err != nil {
		return
	}

	return l.Scale(-1), nil
}
