package loss

import qt "github.com/sahandsafizadeh/qeep/tensor"

type CELoss struct {
}

func NewCE() (c *CELoss) {
	return new(CELoss)
}

func (c *CELoss) Compute(yp qt.Tensor, yt qt.Tensor) (l qt.Tensor, err error) {
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
