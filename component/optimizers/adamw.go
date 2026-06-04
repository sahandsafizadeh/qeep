package optimizers

import (
	"fmt"
	"math"

	"github.com/sahandsafizadeh/qeep/tensor"
)

// AdamW implements Adam with decoupled weight decay regularization.
type AdamW struct {
	learningRate float64
	weightDecay  float64
	beta1        float64
	beta2        float64
	eps          float64
	iteration    map[*tensor.Tensor]int
	velocities1  map[*tensor.Tensor]tensor.Tensor
	velocities2  map[*tensor.Tensor]tensor.Tensor
}

type AdamWConfig struct {
	LearningRate float64
	WeightDecay  float64
	Beta1        float64
	Beta2        float64
	Eps          float64
}

const (
	AdamWDefaultLearningRate = 0.001
	AdamWDefaultWeightDecay  = 0.01
	AdamWDefaultBeta1        = 0.9
	AdamWDefaultBeta2        = 0.999
	AdamWDefaultEps          = 1e-8
)

func NewAdamW(conf *AdamWConfig) (c *AdamW, err error) {
	conf, err = toValidAdamWConfig(conf)
	if err != nil {
		return c, fmt.Errorf("AdamW config data validation failed: %w", err)
	}

	return &AdamW{
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

func (c *AdamW) Update(wptr *tensor.Tensor) (err error) {
	w, g, err := getValidOptimizerInputs(wptr)
	if err != nil {
		return fmt.Errorf("AdamW input data validation failed: %w", err)
	}

	err = c.update(wptr, w, g)
	if err != nil {
		return fmt.Errorf("AdamW update failed: %w", err)
	}

	return nil
}

func (c *AdamW) update(wptr *tensor.Tensor, w tensor.Tensor, g tensor.Tensor) (err error) {
	delta := g

	mh := delta.Scale(1 - c.beta1)
	vh := delta.Pow(2).Scale(1 - c.beta2)

	if m, ok := c.velocities1[wptr]; ok {
		m = m.Scale(c.beta1)

		mh, err = mh.Add(m)
		if err != nil {
			return err
		}
	}

	if v, ok := c.velocities2[wptr]; ok {
		v = v.Scale(c.beta2)

		vh, err = vh.Add(v)
		if err != nil {
			return err
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

	eps, err := c.toUntrackedFull(vh, c.eps)
	if err != nil {
		return err
	}

	d, err := vh.Add(eps)
	if err != nil {
		return err
	}

	delta, err = n.Div(d)
	if err != nil {
		return err
	}

	wdt := w.Scale(c.weightDecay)

	delta, err = delta.Add(wdt)
	if err != nil {
		return err
	}

	delta = delta.Scale(c.learningRate)

	*wptr, err = w.Sub(delta)
	if err != nil {
		return err
	}

	return nil
}

/* ----- helpers ----- */

func (c *AdamW) toUntrackedFull(x tensor.Tensor, value float64) (y tensor.Tensor, err error) {
	dev := x.Device()
	dims := x.Shape()

	return tensor.Full(dims, value, &tensor.Config{
		Device:    dev,
		GradTrack: false,
	})
}

func toValidAdamWConfig(iconf *AdamWConfig) (conf *AdamWConfig, err error) {
	if iconf == nil {
		iconf = &AdamWConfig{
			LearningRate: AdamWDefaultLearningRate,
			WeightDecay:  AdamWDefaultWeightDecay,
			Beta1:        AdamWDefaultBeta1,
			Beta2:        AdamWDefaultBeta2,
			Eps:          AdamWDefaultEps,
		}
	}

	conf = new(AdamWConfig)
	*conf = *iconf

	if conf.LearningRate == 0. {
		conf.LearningRate = AdamWDefaultLearningRate
	}
	if conf.WeightDecay == 0. {
		conf.WeightDecay = AdamWDefaultWeightDecay
	}
	if conf.Beta1 == 0. {
		conf.Beta1 = AdamWDefaultBeta1
	}
	if conf.Beta2 == 0. {
		conf.Beta2 = AdamWDefaultBeta2
	}
	if conf.Eps == 0. {
		conf.Eps = AdamWDefaultEps
	}

	if conf.LearningRate < 0 {
		return conf, fmt.Errorf("expected 'LearningRate' to be positive: got (%f)", conf.LearningRate)
	}

	if conf.WeightDecay < 0 {
		return conf, fmt.Errorf("expected 'WeightDecay' to be positive: got (%f)", conf.WeightDecay)
	}

	if conf.Beta1 < 0 {
		return conf, fmt.Errorf("expected 'Beta1' to be positive: got (%f)", conf.Beta1)
	}

	if conf.Beta2 < 0 {
		return conf, fmt.Errorf("expected 'Beta2' to be positive: got (%f)", conf.Beta2)
	}

	if conf.Eps < 0 {
		return conf, fmt.Errorf("expected 'Eps' to be positive: got (%f)", conf.Eps)
	}

	return conf, nil
}
