package component

import (
	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

type Full struct {
	value float64
}

func (c *Full) Init(shape ...int32) (x qt.Tensor, err error) {
	return tinit.Full(nil, c.value, shape...)
}

// TODO: add uniform(a,b) and normal(u, sigma2) initializers to tinit

// uniform random, normal random, xavior uniform, xavior normal, he uniform, he normal, lecun init
