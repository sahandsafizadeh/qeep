package layers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Input struct {
	SeedFunc func() tensor.Tensor
}

func NewInput() (c *Input) {
	return &Input{}
}

func (c *Input) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	err = c.validateInputs(xs)
	if err != nil {
		err = fmt.Errorf("Input input data validation failed: %w", err)
		return
	}

	return c.SeedFunc(), nil
}

/* ----- helpers ----- */

func (c *Input) validateInputs(xs []tensor.Tensor) (err error) {
	if len(xs) != 0 {
		err = fmt.Errorf("expected no input tensors: got (%d)", len(xs))
		return
	}

	return nil
}
