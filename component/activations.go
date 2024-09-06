package component

import qt "github.com/sahandsafizadeh/qeep/tensor"

type Tanh struct {
}

func (c *Tanh) Trigger(x qt.Tensor) (y qt.Tensor, err error) {
	return x.Tanh(), nil
}

func NewTanh() *Tanh {
	return &Tanh{}
}

type Sigmoid struct {
}

func (c *Sigmoid) Forward(x qt.Tensor) (y qt.Tensor, err error) {
	_1 := x.Pow(0)
	x = x.Scale(-1)
	x = x.Exp()

	y, err = _1.Add(x)
	if err != nil {
		return
	}

	return y.Pow(-1), nil
}

type Softmax struct {
	dim int32
}

func (c *Softmax) Forward(x qt.Tensor) (y qt.Tensor, err error) {
	x = x.Exp()

	s, err := x.SumAlong(c.dim)
	if err != nil {
		return
	}

	return x.Div(s)
}

type Relu struct {
}

func (c *Relu) Forward(x qt.Tensor) (y qt.Tensor, err error) {
	_0 := x.Scale(0)

	return _0.ElMax(x)
}

type LeakyRelu struct {
	m float64 // default = 0.01
}

func (c *LeakyRelu) Forward(x qt.Tensor) (y qt.Tensor, err error) {
	_0 := x.Scale(0)

	s1, err := _0.ElMax(x)
	if err != nil {
		return
	}

	s2, err := _0.ElMin(x)
	if err != nil {
		return
	}

	s2 = s2.Scale(c.m)

	return s1.Add(s2)
}
