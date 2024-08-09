package component

import qt "github.com/sahandsafizadeh/qeep/tensor"

type MSE struct {
}

func (c *MSE) Forward(yp, yt qt.Tensor) (l qt.Tensor, err error) {
	l, err = yt.Sub(yp)
	if err != nil {
		return
	}

	l = l.Pow(2)

	return l.MeanAlong(0)
}

type BinaryCrossEntropy struct {
}

func (c *BinaryCrossEntropy) Forward(yp, yt qt.Tensor) (y qt.Tensor, err error) {
	// 1D inputs, each for a batch entry, output of a sigmoid

	_1 := yp.Pow(0)

	yp1 := yp.Log()

	a, err := yt.Mul(yp1)
	if err != nil {
		return
	}

	x, err := _1.Sub(yp)
	if err != nil {
		return
	}

	t, err := _1.Sub(yt)
	if err != nil {
		return
	}

	b, err := x.Mul(t)
	if err != nil {
		return
	}

	y, err = a.Add(b)
	if err != nil {
		return
	}

	y, err = y.MeanAlong(0)
	if err != nil {
		return
	}

	return y.Scale(-1), nil
}

type CrossEntropy struct {
}

func (c *CrossEntropy) Forward(yp, yt qt.Tensor) (y qt.Tensor, err error) {
	// 2D inputs, first dim for batch and second dim, output of a softmax probability for each class
	yp = yp.Log()

	y, err = yt.Mul(yp)
	if err != nil {
		return
	}

	y, err = y.SumAlong(1)
	if err != nil {
		return
	}

	y, err = y.MeanAlong(0)
	if err != nil {
		return
	}

	return y.Scale(-1), nil
}
