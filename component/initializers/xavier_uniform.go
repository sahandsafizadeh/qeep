package initializers

import (
	"fmt"
	"math"

	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

type XavierUniform struct {
	fanIn  int32
	fanOut int32
}

type XavierUniformConfig struct {
	FanIn  int32
	FanOut int32
}

func NewXavierUniform(conf *XavierUniformConfig) (c *XavierUniform, err error) {
	conf, err = toValidXavierUniformConfig(conf)
	if err != nil {
		return
	}

	return &XavierUniform{
		fanIn:  conf.FanIn,
		fanOut: conf.FanOut,
	}, nil
}

func (c *XavierUniform) Init(dev tinit.Device, shape []int32) (x qt.Tensor, err error) {
	r := math.Sqrt(6. / float64(c.fanIn+c.fanOut))
	return tinit.RandU(tensorConf(dev), -r, r, shape...)
}

/* ----- helpers ----- */

func toValidXavierUniformConfig(iconf *XavierUniformConfig) (conf *XavierUniformConfig, err error) {
	if iconf == nil {
		err = fmt.Errorf("expected xavier uniform config not to be nil")
		return
	}

	conf = new(XavierUniformConfig)
	*conf = *iconf

	if conf.FanIn <= 0 {
		err = fmt.Errorf("expected xavier uniform 'FanIn' to be positive: got (%d)", conf.FanIn)
		return
	}

	if conf.FanOut <= 0 {
		err = fmt.Errorf("expected xavier uniform 'FanOut' to be positive: got (%d)", conf.FanOut)
		return
	}

	return conf, nil
}
