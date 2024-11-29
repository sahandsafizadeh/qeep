package optimizers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type SGD struct {
	learningRate float64
}

type SGDConfig struct {
	LearningRate float64
}

const sgdDefaultLearningRate = 0.01

func NewSGD(conf *SGDConfig) (c *SGD) {
	conf = toValidSGDConfig(conf)

	return &SGD{
		learningRate: conf.LearningRate,
	}
}

func (c *SGD) Update(wptr *tensor.Tensor) (err error) {
	w, g, err := c.toValidInputs(wptr)
	if err != nil {
		err = fmt.Errorf("SGD input data validation failed: %w", err)
		return
	}

	delta := g.Scale(c.learningRate)

	*wptr, err = w.Sub(delta)
	if err != nil {
		return
	}

	return nil
}

/* ----- helpers ----- */

func (c *SGD) toValidInputs(wptr *tensor.Tensor) (w tensor.Tensor, g tensor.Tensor, err error) {
	if wptr == nil {
		err = fmt.Errorf("expected tensor's pointer not to be nil")
		return
	}

	w = *wptr
	if w == nil {
		err = fmt.Errorf("expected tensor not to be nil")
		return
	}

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
		}
	}

	conf = new(SGDConfig)
	*conf = *iconf

	return conf
}
