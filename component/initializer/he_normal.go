package initializers

import (
	"fmt"
	"math"

	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

type HeNormal struct {
	fanIn int32
}

type HeNormalConfig struct {
	FanIn int32
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

func (c *HeNormal) Init(shape []int32) (x qt.Tensor, err error) {
	s := math.Sqrt(2. / float64(c.fanIn))
	return tinit.RandN(tensorInitConf(), 0., s, shape...)
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
