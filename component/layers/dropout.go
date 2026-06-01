package layers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

// Dropout randomly zeros elements with probability Rate during training; no-op when gradients are disabled.
type Dropout struct {
	rate float64
}

type DropoutConfig struct {
	Rate float64
}

const DropoutDefaultRate = 0.5

func NewDropout(conf *DropoutConfig) (c *Dropout, err error) {
	conf, err = toValidDropoutConfig(conf)
	if err != nil {
		return c, fmt.Errorf("Dropout config data validation failed: %w", err)
	}

	return &Dropout{
		rate: conf.Rate,
	}, nil
}

func (c *Dropout) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	x, err := c.toValidInputs(xs)
	if err != nil {
		return y, fmt.Errorf("Dropout input data validation failed: %w", err)
	}

	y, err = c.forward(x)
	if err != nil {
		return y, fmt.Errorf("Dropout forward failed: %w", err)
	}

	return y, nil
}

func (c *Dropout) forward(x tensor.Tensor) (y tensor.Tensor, err error) {
	if !x.GradientTracked() {
		return x, nil
	}

	rate := c.rate
	scale := 1 / (1 - rate)

	dev := x.Device()
	dims := x.Shape()

	droRate, err := tensor.Full(dims, rate, &tensor.Config{
		Device:    dev,
		GradTrack: false,
	})
	if err != nil {
		return y, err
	}

	droProb, err := tensor.RandU(dims, 0, 1, &tensor.Config{
		Device:    dev,
		GradTrack: false,
	})
	if err != nil {
		return y, err
	}

	dropout, err := droRate.Le(droProb)
	if err != nil {
		return y, err
	}

	dropout = dropout.Scale(scale)

	return x.Mul(dropout)
}

/* ----- helpers ----- */

func (c *Dropout) toValidInputs(xs []tensor.Tensor) (x tensor.Tensor, err error) {
	if len(xs) != 1 {
		return x, fmt.Errorf("expected exactly one input tensor: got (%d)", len(xs))
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

	if conf.Rate < 0 || conf.Rate >= 1 {
		return conf, fmt.Errorf("expected 'Rate' to be in range [0,1): got (%f)", conf.Rate)
	}

	return conf, nil
}
