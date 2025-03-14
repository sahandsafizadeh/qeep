package optimizers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type SGD struct {
	learningRate float64
	momentum     float64
	velocities   map[*tensor.Tensor]tensor.Tensor
}

type SGDConfig struct {
	LearningRate float64
	Momentum     float64
}

const (
	sgdDefaultLearningRate = 0.01
	sgdDefaultMomentum     = 0.
)

func NewSGD(conf *SGDConfig) (c *SGD) {
	conf = toValidSGDConfig(conf)

	c = &SGD{
		learningRate: conf.LearningRate,
		momentum:     conf.Momentum,
	}

	if c.hasMomentum() {
		c.velocities = make(map[*tensor.Tensor]tensor.Tensor)
	}

	return c
}

func (c *SGD) Update(wptr *tensor.Tensor) (err error) {
	w, g, err := c.toValidInputs(wptr)
	if err != nil {
		err = fmt.Errorf("SGD input data validation failed: %w", err)
		return
	}

	delta := g.Scale(c.learningRate)

	if c.hasMomentum() {
		if v, ok := c.velocities[wptr]; ok {
			mmt := v.Scale(c.momentum)

			delta, err = mmt.Add(delta)
			if err != nil {
				return
			}
		}

		c.velocities[wptr] = delta
	}

	*wptr, err = w.Sub(delta)
	if err != nil {
		return
	}

	return nil
}

func (c *SGD) hasMomentum() (has bool) {
	return c.momentum != 0.
}

/* ----- helpers ----- */

func (c *SGD) toValidInputs(wptr *tensor.Tensor) (w tensor.Tensor, g tensor.Tensor, err error) {
	w = *wptr
	g = w.Gradient()

	if g == nil {
		err = fmt.Errorf("expected tensor's gradient not to be nil")
		return
	}

	return w, g, nil
}

func toValidSGDConfig(iconf *SGDConfig) (conf *SGDConfig) {
	if iconf == nil {
		iconf = &SGDConfig{
			LearningRate: sgdDefaultLearningRate,
			Momentum:     sgdDefaultMomentum,
		}
	}

	conf = new(SGDConfig)
	*conf = *iconf

	return conf
}
