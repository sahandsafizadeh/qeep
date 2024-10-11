package initializer

import (
	"fmt"
	"math"

	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
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
		return
	}

	return &XavierNormal{
		fanIn:  conf.FanIn,
		fanOut: conf.FanOut,
	}, nil
}

func (c *XavierNormal) Init(shape []int) (x qt.Tensor, err error) {
	s := math.Sqrt(2. / float64(c.fanIn+c.fanOut))
	return tinit.RandN(shape, 0., s, tensorInitConf())
}

/* ----- helpers ----- */

func toValidXavierNormalConfig(iconf *XavierNormalConfig) (conf *XavierNormalConfig, err error) {
	if iconf == nil {
		err = fmt.Errorf("expected xavier normal config not to be nil")
		return
	}

	conf = new(XavierNormalConfig)
	*conf = *iconf

	if conf.FanIn <= 0 {
		err = fmt.Errorf("expected xavier normal 'FanIn' to be positive: got (%d)", conf.FanIn)
		return
	}

	if conf.FanOut <= 0 {
		err = fmt.Errorf("expected xavier normal 'FanOut' to be positive: got (%d)", conf.FanOut)
		return
	}

	return conf, nil
}
