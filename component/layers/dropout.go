package layers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Dropout struct {
	rate float64
}

type DropoutConfig struct {
	Rate float64
}

const DropoutDefaultRate = 0.5

func NewDropout(conf *DropoutConfig) (c *Dropout) {
	conf = toValidDropoutConfig(conf)

	return &Dropout{
		rate: conf.Rate,
	}
}

func (c *Dropout) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	x, err := c.toValidInputs(xs)
	if err != nil {
		err = fmt.Errorf("Dropout input data validation failed: %w", err)
		return
	}

	return c.forward(x)
}

func (c *Dropout) forward(x tensor.Tensor) (y tensor.Tensor, err error) {
	// todo: use in proper time
	if x.Gradient() == nil {
		return x, nil
	}

	p, err := tensor.RandU(x.Shape(), 0, 1, &tensor.Config{
		GradTrack: false,
	})
	if err != nil {
		return
	}

	q, err := tensor.Full(x.Shape(), c.rate, &tensor.Config{
		GradTrack: false,
	})
	if err != nil {
		return
	}

	r, err := p.Le(q)
	if err != nil {
		return
	}

	y, err = x.Mul(r)
	if err != nil {
		return
	}

	scale := 1. / (1 - c.rate)

	y = y.Scale(scale)

	return y, nil
}

/* ----- helpers ----- */

func (c *Dropout) toValidInputs(xs []tensor.Tensor) (x tensor.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected exactly one input tensor: got (%d)", len(xs))
		return
	}

	x = xs[0]

	return x, nil
}

func toValidDropoutConfig(iconf *DropoutConfig) (conf *DropoutConfig, err error) {
	if iconf == nil {
		iconf = &DropoutConfig{
			Rate: DropoutDefaultRate,
		}
	}

	conf = new(DropoutConfig)
	*conf = *iconf

	if conf.Rate < 0 {
		err = fmt.Errorf("expected 'Dim' not to be negative: got (%d)", conf.Dim)
		return
	}

	return conf, nil
}
