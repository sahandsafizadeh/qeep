package initializers

import (
	"fmt"
	"math"

	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

type HeUniform struct {
	fanIn int32
}

type HeUniformConfig struct {
	FanIn int32
}

func NewHeUniform(conf *HeUniformConfig) (c *HeUniform, err error) {
	conf, err = toValidHeUniformConfig(conf)
	if err != nil {
		return
	}

	return &HeUniform{
		fanIn: conf.FanIn,
	}, nil
}

func (c *HeUniform) Init(conf *tinit.Config, shape []int32) (x qt.Tensor, err error) {
	r := math.Sqrt(6. / float64(c.fanIn))
	return tinit.RandU(conf, -r, r, shape...)
}

/* ----- helpers ----- */

func toValidHeUniformConfig(iconf *HeUniformConfig) (conf *HeUniformConfig, err error) {
	if iconf == nil {
		err = fmt.Errorf("expected he uniform config not to be nil")
		return
	}

	conf = new(HeUniformConfig)
	*conf = *iconf

	if conf.FanIn <= 0 {
		err = fmt.Errorf("expected he uniform 'FanIn' to be positive: got (%d)", conf.FanIn)
		return
	}

	return conf, nil
}
