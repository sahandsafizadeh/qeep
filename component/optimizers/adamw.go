package optimizers

import (
	"fmt"
	"math"

	"github.com/sahandsafizadeh/qeep/tensor"
)

// AdamW implements Adam with decoupled weight decay regularization.
// Unlike standard Adam, weight decay is applied directly rather than through gradient.
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

// AdamWConfig specifies learning rate, decoupled weight decay, and moment coefficients.
// Zero values are replaced with package defaults (see AdamWDefault* constants).
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

// NewAdamW creates an AdamW optimizer. conf may be nil; then defaults are used.
func NewAdamW(conf *AdamWConfig) (c *AdamW, err error) {
	conf, err = toValidAdamWConfig(conf)
	if err != nil {
		err = fmt.Errorf("AdamW config data validation failed: %w", err)
		return
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
		err = fmt.Errorf("AdamW input data validation failed: %w", err)
		return
	}

	delta := g

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

	eps, err := c.toUntrackedFull(vh, c.eps)
	if err != nil {
		return
	}

	d, err := vh.Add(eps)
	if err != nil {
		return
	}

	delta, err = n.Div(d)
	if err != nil {
		return
	}

	wdt := w.Scale(c.weightDecay)

	delta, err = delta.Add(wdt)
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
