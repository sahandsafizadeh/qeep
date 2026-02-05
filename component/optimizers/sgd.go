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

// SGDConfig specifies learning rate, L2 weight decay, and momentum coefficient.
// Zero values are replaced with package defaults (see SGDDefault* constants).
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

// NewSGD creates an SGD optimizer. conf may be nil; then defaults are used.
func NewSGD(conf *SGDConfig) (c *SGD, err error) {
	conf, err = toValidSGDConfig(conf)
	if err != nil {
		err = fmt.Errorf("SGD config data validation failed: %w", err)
		return
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
	return c.weightDecay > 0
}

func (c *SGD) hasMomentum() (has bool) {
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
		err = fmt.Errorf("expected 'LearningRate' to be positive: got (%f)", conf.LearningRate)
		return
	}

	if conf.WeightDecay < 0 {
		err = fmt.Errorf("expected 'WeightDecay' not to be negative: got (%f)", conf.WeightDecay)
		return
	}

	if conf.Momentum < 0 {
		err = fmt.Errorf("expected 'Momentum' not to be negative: got (%f)", conf.Momentum)
		return
	}

	return conf, nil
}
