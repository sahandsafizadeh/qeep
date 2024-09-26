package losses

import qt "github.com/sahandsafizadeh/qeep/tensor"

func MSELoss(yp qt.Tensor, yt qt.Tensor) (l qt.Tensor, err error) {
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
