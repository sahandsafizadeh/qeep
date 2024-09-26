package loss

import qt "github.com/sahandsafizadeh/qeep/tensor"

type MSELoss struct {
}

func NewMSELoss() (c *MSELoss) {
	return new(MSELoss)
}

func (c *MSELoss) Compute(yp qt.Tensor, yt qt.Tensor) (l qt.Tensor, err error) {
	l, err = yt.Sub(yp)
	if err != nil {
		return
	}

	l = l.Pow(2)

	l, err = l.MeanAlong(0)
	if err != nil {
		return
	}

	return l, nil
}
