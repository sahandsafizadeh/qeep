package initializers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Full struct {
	value float64
}

type FullConfig struct {
	Value float64
}

const FullDefaultValue = 0.

func NewFull(conf *FullConfig) *Full {
	conf = toValidFullConfig(conf)

	return &Full{
		value: conf.Value,
	}
}

func (c *Full) Init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	x, err = c.init(shape, device)
	if err != nil {
		return x, fmt.Errorf("Full init failed: %w", err)
	}

	return x, nil
}

func (c *Full) init(shape []int, device tensor.Device) (x tensor.Tensor, err error) {
	return tensor.Full(shape, c.value, tensorInitConf(device))
}

/* ----- helpers ----- */

func toValidFullConfig(iconf *FullConfig) *FullConfig {
	if iconf == nil {
		iconf = &FullConfig{
			Value: FullDefaultValue,
		}
	}

	conf := new(FullConfig)
	*conf = *iconf

	return conf
}
