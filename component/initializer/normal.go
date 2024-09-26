package initializer

import (
	"fmt"

	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
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
		return
	}

	return &Normal{
		mean:   conf.Mean,
		stdDev: conf.StdDev,
	}, nil
}

func (c *Normal) Init(shape []int32) (x qt.Tensor, err error) {
	return tinit.RandN(tensorInitConf(), c.mean, c.stdDev, shape...)
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
		err = fmt.Errorf("expected normal 'StdDev' to be positive: got (%f)", conf.StdDev)
		return
	}

	return conf, nil
}
