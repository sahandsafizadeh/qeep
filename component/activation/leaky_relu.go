package component

import qt "github.com/sahandsafizadeh/qeep/tensor"

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
