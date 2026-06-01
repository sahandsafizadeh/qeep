package initializers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Normal struct {
	mean   float64
	stdDev float64
}

type NormalConfig struct {
	Mean   float64
	StdDev float64
}

const NormalDefaultMean = 0.
const NormalDefaultStdDev = 0.05

func NewNormal(conf *NormalConfig) (c *Normal, err error) {
	conf, err = toValidNormalConfig(conf)
	if err != nil {
		return c, fmt.Errorf("Normal config data validation failed: %w", err)
	}

	return &Normal{
		mean:   conf.Mean,
		stdDev: conf.StdDev,
	}, nil
}

func (c *Normal) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	x, err = c.init(shape, device)
	if err != nil {
		return x, fmt.Errorf("Normal init failed: %w", err)
	}

	return x, nil
}

func (c *Normal) init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	return tensor.RandN(shape, c.mean, c.stdDev, tensorInitConf(device))
}

/* ----- helpers ----- */

func toValidNormalConfig(iconf *NormalConfig) (conf *NormalConfig, err error) {
	if iconf == nil {
		iconf = &NormalConfig{
			Mean:   NormalDefaultMean,
			StdDev: NormalDefaultStdDev,
		}
	}

	conf = new(NormalConfig)
	*conf = *iconf

	if !(conf.StdDev > 0) {
		return conf, fmt.Errorf("expected 'StdDev' to be positive: got (%f)", conf.StdDev)
	}

	return conf, nil
}
