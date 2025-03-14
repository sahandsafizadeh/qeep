package initializers

import "github.com/sahandsafizadeh/qeep/tensor"

type Full struct {
	value float64
}

type FullConfig struct {
	Value float64
}

const FullDefaultValue = 0.

func NewFull(conf *FullConfig) (c *Full) {
	conf = toValidFullConfig(conf)

	return &Full{
		value: conf.Value,
	}
}

func (c *Full) Init(shape []int) (x tensor.Tensor, err error) {
	return tensor.Full(shape, c.value, tensorInitConf())
}

/* ----- helpers ----- */

func toValidFullConfig(iconf *FullConfig) (conf *FullConfig) {
	if iconf == nil {
		iconf = &FullConfig{
			Value: FullDefaultValue,
		}
	}

	conf = new(FullConfig)
	*conf = *iconf

	return conf
}
