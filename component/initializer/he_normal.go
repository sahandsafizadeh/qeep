package initializer

import (
	"fmt"
	"math"

	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

type HeNormal struct {
	fanIn int
}

type HeNormalConfig struct {
	FanIn int
}

func NewHeNormal(conf *HeNormalConfig) (c *HeNormal, err error) {
	conf, err = toValidHeNormalConfig(conf)
	if err != nil {
		return
	}

	return &HeNormal{
		fanIn: conf.FanIn,
	}, nil
}

func (c *HeNormal) Init(shape []int) (x qt.Tensor, err error) {
	s := math.Sqrt(2. / float64(c.fanIn))
	return tinit.RandN(shape, 0., s, tensorInitConf())
}

/* ----- helpers ----- */

func toValidHeNormalConfig(iconf *HeNormalConfig) (conf *HeNormalConfig, err error) {
	if iconf == nil {
		err = fmt.Errorf("expected he normal config not to be nil")
		return
	}

	conf = new(HeNormalConfig)
	*conf = *iconf

	if conf.FanIn <= 0 {
		err = fmt.Errorf("expected he normal 'FanIn' to be positive: got (%d)", conf.FanIn)
		return
	}

	return conf, nil
}
