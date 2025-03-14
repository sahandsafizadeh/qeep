package initializers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Uniform struct {
	lower float64
	upper float64
}

type UniformConfig struct {
	Lower float64
	Upper float64
}

const UniformDefaultLower = -0.05
const UniformDefaultUpper = 0.05

func NewUniform(conf *UniformConfig) (c *Uniform, err error) {
	conf, err = toValidUniformConfig(conf)
	if err != nil {
		err = fmt.Errorf("Uniform config data validation failed: %w", err)
		return
	}

	return &Uniform{
		lower: conf.Lower,
		upper: conf.Upper,
	}, nil
}

func (c *Uniform) Init(shape []int) (x tensor.Tensor, err error) {
	return tensor.RandU(shape, c.lower, c.upper, tensorInitConf())
}

/* ----- helpers ----- */

func toValidUniformConfig(iconf *UniformConfig) (conf *UniformConfig, err error) {
	if iconf == nil {
		iconf = &UniformConfig{
			Lower: UniformDefaultLower,
			Upper: UniformDefaultUpper,
		}
	}

	conf = new(UniformConfig)
	*conf = *iconf

	if !(conf.Lower < conf.Upper) {
		err = fmt.Errorf("expected 'Lower' to be less than 'Upper': (%f) >= (%f)", conf.Lower, conf.Upper)
		return
	}

	return conf, nil
}
