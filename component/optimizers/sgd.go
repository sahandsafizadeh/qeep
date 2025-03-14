package optimizers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type SGD struct {
	learningRate float64
	weightDecay  float64
	momentum     float64
	velocities   map[*tensor.Tensor]tensor.Tensor
}

type SGDConfig struct {
	LearningRate float64
	WeightDecay  float64
	Momentum     float64
}

const (
	sgdDefaultLearningRate = 0.01
	sgdDefaultWeightDecay  = 0.
	sgdDefaultMomentum     = 0.
)

func NewSGD(conf *SGDConfig) (c *SGD) {
	conf = toValidSGDConfig(conf)

	c = &SGD{
		learningRate: conf.LearningRate,
		weightDecay:  conf.WeightDecay,
		momentum:     conf.Momentum,
	}

	if c.hasMomentum() {
		c.velocities = make(map[*tensor.Tensor]tensor.Tensor)
	}

	return c
}

func (c *SGD) Update(wptr *tensor.Tensor) (err error) {
	w, g, err := getValidOptimizerInputs(wptr)
	if err != nil {
		err = fmt.Errorf("SGD input data validation failed: %w", err)
		return
	}

	delta := g

	if c.hasWeightDecay() {
		wdt := w.Scale(c.weightDecay)

		delta, err = delta.Add(wdt)
		if err != nil {
			return
		}
	}

	if c.hasMomentum() {
		if v, ok := c.velocities[wptr]; ok {
			mmt := v.Scale(c.momentum)

			delta, err = delta.Add(mmt)
			if err != nil {
				return
			}
		}

		c.velocities[wptr] = delta
	}

	delta = delta.Scale(c.learningRate)

	*wptr, err = w.Sub(delta)
	if err != nil {
		return
	}

	return nil
}

func (c *SGD) hasWeightDecay() (has bool) {
	return c.weightDecay != 0.
}

func (c *SGD) hasMomentum() (has bool) {
	return c.momentum != 0.
}

/* ----- helpers ----- */

func toValidSGDConfig(iconf *SGDConfig) (conf *SGDConfig) {
	if iconf == nil {
		iconf = &SGDConfig{
			LearningRate: sgdDefaultLearningRate,
			WeightDecay:  sgdDefaultWeightDecay,
			Momentum:     sgdDefaultMomentum,
		}
	}

	conf = new(SGDConfig)
	*conf = *iconf

	return conf
}
