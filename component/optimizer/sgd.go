package optimizer

import (
	"fmt"

	qt "github.com/sahandsafizadeh/qeep/tensor"
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

func (c *SGD) Update(wptr *qt.Tensor) (err error) {
	if wptr == nil {
		err = fmt.Errorf("expected weight pointer not to be nil")
		return
	}

	w := *wptr
	g := w.Gradient()

	if g == nil {
		err = fmt.Errorf("weight is not trainable: expected weight gradient not to be nil")
		return
	}

	*wptr, err = w.Sub(g.Scale(c.learningRate))
	if err != nil {
		return
	}

	return nil
}

/* ----- helpers ----- */

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
