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
		return nil, fmt.Errorf("HeUniform config data validation failed: %w", err)
	}

	return &HeUniform{
		fanIn: conf.FanIn,
	}, nil
}

func (c *HeUniform) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	x, err = c.init(shape, device)
	if err != nil {
		return x, fmt.Errorf("HeUniform init failed: %w", err)
	}

	return x, nil
}

func (c *HeUniform) init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	r := math.Sqrt(6. / float64(c.fanIn))
	return tensor.RandU(shape, -r, r, tensorInitConf(device))
}

/* ----- helpers ----- */

func toValidHeUniformConfig(iconf *HeUniformConfig) (conf *HeUniformConfig, err error) {
	if iconf == nil {
		return nil, fmt.Errorf("expected config not to be nil")
	}

	conf = new(HeUniformConfig)
	*conf = *iconf

	if conf.FanIn <= 0 {
		return nil, fmt.Errorf("expected 'FanIn' to be positive: got (%d)", conf.FanIn)
	}

	return conf, nil
}
