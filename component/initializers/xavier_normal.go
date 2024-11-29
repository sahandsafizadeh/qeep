package initializers

import (
	"fmt"
	"math"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type XavierNormal struct {
	fanIn  int
	fanOut int
}

type XavierNormalConfig struct {
	FanIn  int
	FanOut int
}

func NewXavierNormal(conf *XavierNormalConfig) (c *XavierNormal, err error) {
	conf, err = toValidXavierNormalConfig(conf)
	if err != nil {
		err = fmt.Errorf("XavierNormal config data validation failed: %w", err)
		return
	}

	return &XavierNormal{
		fanIn:  conf.FanIn,
		fanOut: conf.FanOut,
	}, nil
}

func (c *XavierNormal) Init(shape []int) (x tensor.Tensor, err error) {
	s := math.Sqrt(2. / float64(c.fanIn+c.fanOut))
	return tensor.RandN(shape, 0., s, tensorInitConf())
}

/* ----- helpers ----- */

func toValidXavierNormalConfig(iconf *XavierNormalConfig) (conf *XavierNormalConfig, err error) {
	if iconf == nil {
		err = fmt.Errorf("expected config not to be nil")
		return
	}

	conf = new(XavierNormalConfig)
	*conf = *iconf

	if conf.FanIn <= 0 {
		err = fmt.Errorf("expected 'FanIn' to be positive: got (%d)", conf.FanIn)
		return
	}

	if conf.FanOut <= 0 {
		err = fmt.Errorf("expected 'FanOut' to be positive: got (%d)", conf.FanOut)
		return
	}

	return conf, nil
}
