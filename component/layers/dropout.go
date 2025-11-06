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

func NewDropout(conf *DropoutConfig) (c *Dropout, err error) {
	conf, err = toValidDropoutConfig(conf)
	if err != nil {
		err = fmt.Errorf("Dropout config data validation failed: %w", err)
		return
	}

	return &Dropout{
		rate: conf.Rate,
	}, nil
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
		return
	}

	droProb, err := tensor.RandU(dims, 0, 1, &tensor.Config{
		Device:    dev,
		GradTrack: false,
	})
	if err != nil {
		return
	}

	dropout, err := droRate.Le(droProb)
	if err != nil {
		return
	}

	dropout = dropout.Scale(scale)

	return x.Mul(dropout)
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

	if conf.Rate < 0 || conf.Rate >= 1 {
		err = fmt.Errorf("expected 'Rate' to be in range [0,1): got (%f)", conf.Rate)
		return
	}

	return conf, nil
}
