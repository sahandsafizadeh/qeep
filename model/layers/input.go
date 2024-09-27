package layers

import qt "github.com/sahandsafizadeh/qeep/tensor"

type input struct {
	SeedFunc func() qt.Tensor
}

func NewInput() (c *input) {
	return &input{}
}

func (c *input) Forward(xs ...qt.Tensor) (y qt.Tensor, err error) {
	return c.SeedFunc(), nil
}
