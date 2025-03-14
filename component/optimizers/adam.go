package optimizers

import (
	"fmt"
	"math"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Adam struct {
	learningRate float64
	weightDecay  float64
	beta1        float64
	beta2        float64
	eps          float64
	iteration    map[*tensor.Tensor]int
	velocities1  map[*tensor.Tensor]tensor.Tensor
	velocities2  map[*tensor.Tensor]tensor.Tensor
}

type AdamConfig struct {
	LearningRate float64
	WeightDecay  float64
	Beta1        float64
	Beta2        float64
	Eps          float64
}

const (
	adamDefaultLearningRate = 0.001
	adamDefaultWeightDecay  = 0.
	adamDefaultBeta1        = 0.9
	adamDefaultBeta2        = 0.999
	adamDefaultEps          = 1e-8
)

func NewAdam(conf *AdamConfig) (c *Adam, err error) {
	conf, err = toValidAdamConfig(conf)
	if err != nil {
		err = fmt.Errorf("Adam config data validation failed: %w", err)
		return
	}

	return &Adam{
		learningRate: conf.LearningRate,
		weightDecay:  conf.WeightDecay,
		beta1:        conf.Beta1,
		beta2:        conf.Beta2,
		eps:          conf.Eps,
		iteration:    make(map[*tensor.Tensor]int),
		velocities1:  make(map[*tensor.Tensor]tensor.Tensor),
		velocities2:  make(map[*tensor.Tensor]tensor.Tensor),
	}, nil
}

func (c *Adam) Update(wptr *tensor.Tensor) (err error) {
	w, g, err := getValidOptimizerInputs(wptr)
	if err != nil {
		err = fmt.Errorf("Adam input data validation failed: %w", err)
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

	mh := delta.Scale(1 - c.beta1)
	vh := delta.Pow(2).Scale(1 - c.beta2)

	if m, ok := c.velocities1[wptr]; ok {
		m = m.Scale(c.beta1)

		mh, err = mh.Add(m)
		if err != nil {
			return
		}
	}

	if v, ok := c.velocities2[wptr]; ok {
		v = v.Scale(c.beta2)

		vh, err = vh.Add(v)
		if err != nil {
			return
		}
	}

	c.velocities1[wptr] = mh
	c.velocities2[wptr] = vh
	c.iteration[wptr]++

	t := float64(c.iteration[wptr])
	beta1t := math.Pow(c.beta1, t)
	beta2t := math.Pow(c.beta2, t)
	mh = mh.Scale(1 / (1 - beta1t))
	vh = vh.Scale(1 / (1 - beta2t)).Pow(0.5)

	n := mh
	eps := vh.Pow(0).Scale(c.eps)

	d, err := vh.Add(eps)
	if err != nil {
		return
	}

	delta, err = n.Div(d)
	if err != nil {
		return
	}

	delta = delta.Scale(c.learningRate)

	*wptr, err = w.Sub(delta)
	if err != nil {
		return
	}

	return nil
}

func (c *Adam) hasWeightDecay() (has bool) {
	return c.weightDecay > 0
}

/* ----- helpers ----- */

func toValidAdamConfig(iconf *AdamConfig) (conf *AdamConfig, err error) {
	if iconf == nil {
		iconf = &AdamConfig{
			LearningRate: adamDefaultLearningRate,
			WeightDecay:  adamDefaultWeightDecay,
			Beta1:        adamDefaultBeta1,
			Beta2:        adamDefaultBeta2,
			Eps:          adamDefaultEps,
		}
	}

	conf = new(AdamConfig)
	*conf = *iconf

	if conf.LearningRate <= 0 {
		err = fmt.Errorf("expected 'LearningRate' to be positive: got (%f)", conf.LearningRate)
		return
	}

	if conf.WeightDecay < 0 {
		err = fmt.Errorf("expected 'WeightDecay' not to be negative: got (%f)", conf.WeightDecay)
		return
	}

	if conf.Beta1 < 0 {
		err = fmt.Errorf("expected 'Beta1' not to be negative: got (%f)", conf.Beta1)
		return
	}

	if conf.Beta2 < 0 {
		err = fmt.Errorf("expected 'Beta2' not to be negative: got (%f)", conf.Beta2)
		return
	}

	if conf.Eps <= 0 {
		err = fmt.Errorf("expected 'Eps' to be positive: got (%f)", conf.Eps)
		return
	}

	return conf, nil
}
