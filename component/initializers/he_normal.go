package initializers

import (
	"fmt"
	"math"

	"github.com/sahandsafizadeh/qeep/tensor"
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
		return c, fmt.Errorf("HeNormal config data validation failed: %w", err)
	}

	return &HeNormal{
		fanIn: conf.FanIn,
	}, nil
}

func (c *HeNormal) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	x, err = c.init(shape, device)
	if err != nil {
		return x, fmt.Errorf("HeNormal init failed: %w", err)
	}

	return x, nil
}

func (c *HeNormal) init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	s := math.Sqrt(2. / float64(c.fanIn))
	return tensor.RandN(shape, 0., s, tensorInitConf(device))
}

/* ----- helpers ----- */

func toValidHeNormalConfig(iconf *HeNormalConfig) (conf *HeNormalConfig, err error) {
	if iconf == nil {
		return conf, fmt.Errorf("expected config not to be nil")
	}

	conf = new(HeNormalConfig)
	*conf = *iconf

	if conf.FanIn <= 0 {
		return conf, fmt.Errorf("expected 'FanIn' to be positive: got (%d)", conf.FanIn)
	}

	return conf, nil
}
