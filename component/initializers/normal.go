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

const normalDefaultMean = 0.
const normalDefaultStdDev = 0.05

func NewNormal(conf *NormalConfig) (c *Normal, err error) {
	conf, err = toValidNormalConfig(conf)
	if err != nil {
		err = fmt.Errorf("Normal config data validation failed: %w", err)
		return
	}

	return &Normal{
		mean:   conf.Mean,
		stdDev: conf.StdDev,
	}, nil
}

func (c *Normal) Init(shape []int) (x tensor.Tensor, err error) {
	return tensor.RandN(shape, c.mean, c.stdDev, tensorInitConf())
}

/* ----- helpers ----- */

func toValidNormalConfig(iconf *NormalConfig) (conf *NormalConfig, err error) {
	if iconf == nil {
		iconf = &NormalConfig{
			Mean:   normalDefaultMean,
			StdDev: normalDefaultStdDev,
		}
	}

	conf = new(NormalConfig)
	*conf = *iconf

	if !(conf.StdDev > 0) {
		err = fmt.Errorf("expected 'StdDev' to be positive: got (%f)", conf.StdDev)
		return
	}

	return conf, nil
}
