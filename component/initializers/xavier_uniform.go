package initializers

import (
	"fmt"
	"math"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type XavierUniform struct {
	fanIn  int
	fanOut int
}

type XavierUniformConfig struct {
	FanIn  int
	FanOut int
}

func NewXavierUniform(conf *XavierUniformConfig) (c *XavierUniform, err error) {
	conf, err = toValidXavierUniformConfig(conf)
	if err != nil {
		err = fmt.Errorf("XavierUniform config data validation failed: %w", err)
		return
	}

	return &XavierUniform{
		fanIn:  conf.FanIn,
		fanOut: conf.FanOut,
	}, nil
}

func (c *XavierUniform) Init(shape []int) (x tensor.Tensor, err error) {
	r := math.Sqrt(6. / float64(c.fanIn+c.fanOut))
	return tensor.RandU(shape, -r, r, tensorInitConf())
}

/* ----- helpers ----- */

func toValidXavierUniformConfig(iconf *XavierUniformConfig) (conf *XavierUniformConfig, err error) {
	if iconf == nil {
		err = fmt.Errorf("expected config not to be nil")
		return
	}

	conf = new(XavierUniformConfig)
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
