package loss

import qt "github.com/sahandsafizadeh/qeep/tensor"

type MSE struct {
}

func NewMSELoss() (c *MSE) {
	return new(MSE)
}

func (c *MSE) Compute(yp qt.Tensor, yt qt.Tensor) (l qt.Tensor, err error) {
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
