package activation

import qt "github.com/sahandsafizadeh/qeep/tensor"

type Input struct {
	SeedFunc func() qt.Tensor
}

func NewInput() (c *Input) {
	return &Input{}
}

func (c *Input) Forward(xs ...qt.Tensor) (y qt.Tensor, err error) {
	return c.SeedFunc(), nil
}
