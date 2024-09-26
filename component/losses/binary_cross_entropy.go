package losses

import qt "github.com/sahandsafizadeh/qeep/tensor"

func BCELoss(yp qt.Tensor, yt qt.Tensor) (l qt.Tensor, err error) {
	t1 := yt
	y1 := yp

	s1, err := t1.Mul(y1.Log())
	if err != nil {
		return
	}

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

	l, err = s1.Add(s2)
	if err != nil {
		return
	}

	l, err = l.MeanAlong(0)
	if err != nil {
		return
	}

	return l.Scale(-1), nil
}
