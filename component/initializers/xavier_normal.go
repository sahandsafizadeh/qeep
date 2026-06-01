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
		return c, fmt.Errorf("XavierNormal config data validation failed: %w", err)
	}

	return &XavierNormal{
		fanIn:  conf.FanIn,
		fanOut: conf.FanOut,
	}, nil
}

func (c *XavierNormal) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	x, err = c.init(shape, device)
	if err != nil {
		return x, fmt.Errorf("XavierNormal init failed: %w", err)
	}

	return x, nil
}

func (c *XavierNormal) init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	s := math.Sqrt(2. / float64(c.fanIn+c.fanOut))
	return tensor.RandN(shape, 0., s, tensorInitConf(device))
}

/* ----- helpers ----- */

func toValidXavierNormalConfig(iconf *XavierNormalConfig) (conf *XavierNormalConfig, err error) {
	if iconf == nil {
		return conf, fmt.Errorf("expected config not to be nil")
	}

	conf = new(XavierNormalConfig)
	*conf = *iconf

	if conf.FanIn <= 0 {
		return conf, fmt.Errorf("expected 'FanIn' to be positive: got (%d)", conf.FanIn)
	}

	if conf.FanOut <= 0 {
		return conf, fmt.Errorf("expected 'FanOut' to be positive: got (%d)", conf.FanOut)
	}

	return conf, nil
}
