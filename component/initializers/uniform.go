package initializers

import (
	"fmt"

	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

type Uniform struct {
	lower float64
	upper float64
}

type UniformConfig struct {
	Lower float64
	Upper float64
}

const uniformDefaultLower = -0.05
const uniformDefaultUpper = 0.05

func NewUniform(conf *UniformConfig) (c *Uniform, err error) {
	conf, err = toValidUniformConfig(conf)
	if err != nil {
		return
	}

	return &Uniform{
		lower: conf.Lower,
		upper: conf.Upper,
	}, nil
}

func (c *Uniform) Init(dev tinit.Device, shape []int32) (x qt.Tensor, err error) {
	return tinit.RandU(tensorConf(dev), c.lower, c.upper, shape...)
}

/* ----- helpers ----- */

func toValidUniformConfig(iconf *UniformConfig) (conf *UniformConfig, err error) {
	if iconf == nil {
		iconf = &UniformConfig{
			Lower: uniformDefaultLower,
			Upper: uniformDefaultUpper,
		}
	}

	conf = new(UniformConfig)
	*conf = *iconf

	if !(conf.Lower < conf.Upper) {
		err = fmt.Errorf("expected uniform 'Lower' to be less than 'Upper': (%f) >= (%f)", conf.Lower, conf.Upper)
		return
	}

	return conf, nil
}
