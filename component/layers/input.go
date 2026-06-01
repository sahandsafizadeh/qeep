package layers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

// Input is a placeholder layer whose output is set by the model (SeedFunc) when feeding data.
type Input struct {
	SeedFunc func() tensor.Tensor
}

func NewInput() *Input {
	return &Input{}
}

func (c *Input) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	err = c.validateInputs(xs)
	if err != nil {
		return y, fmt.Errorf("Input input data validation failed: %w", err)
	}

	return c.SeedFunc(), nil
}

/* ----- helpers ----- */

func (c *Input) validateInputs(xs []tensor.Tensor) (err error) {
	if len(xs) != 0 {
		return fmt.Errorf("expected no input tensors: got (%d)", len(xs))
	}

	return nil
}
