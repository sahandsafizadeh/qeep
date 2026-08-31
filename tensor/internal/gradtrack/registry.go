package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor/internal/core"

type invertedFullFuncType func(dims []int, value float64, conf *core.Config) (core.Tensor, error)
type invertedTransferFuncType func(t core.Tensor, to core.Device) (core.Tensor, error)

var fullFunc invertedFullFuncType
var transferFunc invertedTransferFuncType

func RegisterFullFunc(fn invertedFullFuncType) {
	if fullFunc != nil {
		panic("gradtrack init: fullFunc has already been registered")
	}

	fullFunc = fn
}

func RegisterTransferFunc(fn invertedTransferFuncType) {
	if transferFunc != nil {
		panic("gradtrack init: transferFunc has already been registered")
	}

	transferFunc = fn
}
