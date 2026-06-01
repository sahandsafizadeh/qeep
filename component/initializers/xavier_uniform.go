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
		return c, fmt.Errorf("XavierUniform config data validation failed: %w", err)
	}

	return &XavierUniform{
		fanIn:  conf.FanIn,
		fanOut: conf.FanOut,
	}, nil
}

func (c *XavierUniform) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	x, err = c.init(shape, device)
	if err != nil {
		return x, fmt.Errorf("XavierUniform init failed: %w", err)
	}

	return x, nil
}

func (c *XavierUniform) init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	r := math.Sqrt(6. / float64(c.fanIn+c.fanOut))
	return tensor.RandU(shape, -r, r, tensorInitConf(device))
}

/* ----- helpers ----- */

func toValidXavierUniformConfig(iconf *XavierUniformConfig) (conf *XavierUniformConfig, err error) {
	if iconf == nil {
		return conf, fmt.Errorf("expected config not to be nil")
	}

	conf = new(XavierUniformConfig)
	*conf = *iconf

	if conf.FanIn <= 0 {
		return conf, fmt.Errorf("expected 'FanIn' to be positive: got (%d)", conf.FanIn)
	}

	if conf.FanOut <= 0 {
		return conf, fmt.Errorf("expected 'FanOut' to be positive: got (%d)", conf.FanOut)
	}

	return conf, nil
}
