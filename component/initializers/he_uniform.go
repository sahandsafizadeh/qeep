package initializers

import (
	"fmt"
	"math"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type HeUniform struct {
	fanIn int
}

type HeUniformConfig struct {
	FanIn int
}

func NewHeUniform(conf *HeUniformConfig) (c *HeUniform, err error) {
	conf, err = toValidHeUniformConfig(conf)
	if err != nil {
		err = fmt.Errorf("HeUniform config data validation failed: %w", err)
		return
	}

	return &HeUniform{
		fanIn: conf.FanIn,
	}, nil
}

func (c *HeUniform) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	r := math.Sqrt(6. / float64(c.fanIn))
	return tensor.RandU(shape, -r, r, tensorInitConf(device))
}

/* ----- helpers ----- */

func toValidHeUniformConfig(iconf *HeUniformConfig) (conf *HeUniformConfig, err error) {
	if iconf == nil {
		err = fmt.Errorf("expected config not to be nil")
		return
	}

	conf = new(HeUniformConfig)
	*conf = *iconf

	if conf.FanIn <= 0 {
		err = fmt.Errorf("expected 'FanIn' to be positive: got (%d)", conf.FanIn)
		return
	}

	return conf, nil
}
