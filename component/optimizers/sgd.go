package optimizers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

// SGD implements stochastic gradient descent with optional momentum and L2 weight decay.
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
	SGDDefaultLearningRate = 0.01
	SGDDefaultWeightDecay  = 0.
	SGDDefaultMomentum     = 0.
)

func NewSGD(conf *SGDConfig) (c *SGD, err error) {
	conf, err = toValidSGDConfig(conf)
	if err != nil {
		return c, fmt.Errorf("SGD config data validation failed: %w", err)
	}

	c = &SGD{
		learningRate: conf.LearningRate,
		weightDecay:  conf.WeightDecay,
		momentum:     conf.Momentum,
	}

	if c.hasMomentum() {
		c.velocities = make(map[*tensor.Tensor]tensor.Tensor)
	}

	return c, nil
}

func (c *SGD) Update(wptr *tensor.Tensor) (err error) {
	w, g, err := getValidOptimizerInputs(wptr)
	if err != nil {
		return fmt.Errorf("SGD input data validation failed: %w", err)
	}

	err = c.update(wptr, w, g)
	if err != nil {
		return fmt.Errorf("SGD update failed: %w", err)
	}

	return nil
}

func (c *SGD) update(wptr *tensor.Tensor, w tensor.Tensor, g tensor.Tensor) (err error) {
	delta := g

	if c.hasWeightDecay() {
		wdt := w.Scale(c.weightDecay)

		delta, err = delta.Add(wdt)
		if err != nil {
			return err
		}
	}

	if c.hasMomentum() {
		if v, ok := c.velocities[wptr]; ok {
			mmt := v.Scale(c.momentum)

			delta, err = delta.Add(mmt)
			if err != nil {
				return err
			}
		}

		c.velocities[wptr] = delta
	}

	delta = delta.Scale(c.learningRate)

	*wptr, err = w.Sub(delta)
	if err != nil {
		return err
	}

	return nil
}

func (c *SGD) hasWeightDecay() bool {
	return c.weightDecay > 0
}

func (c *SGD) hasMomentum() bool {
	return c.momentum > 0
}

/* ----- helpers ----- */

func toValidSGDConfig(iconf *SGDConfig) (conf *SGDConfig, err error) {
	if iconf == nil {
		iconf = &SGDConfig{
			LearningRate: SGDDefaultLearningRate,
			WeightDecay:  SGDDefaultWeightDecay,
			Momentum:     SGDDefaultMomentum,
		}
	}

	conf = new(SGDConfig)
	*conf = *iconf

	if conf.LearningRate <= 0 {
		return conf, fmt.Errorf("expected 'LearningRate' to be positive: got (%f)", conf.LearningRate)
	}

	if conf.WeightDecay < 0 {
		return conf, fmt.Errorf("expected 'WeightDecay' not to be negative: got (%f)", conf.WeightDecay)
	}

	if conf.Momentum < 0 {
		return conf, fmt.Errorf("expected 'Momentum' not to be negative: got (%f)", conf.Momentum)
	}

	return conf, nil
}
