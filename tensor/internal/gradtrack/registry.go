package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor/internal/core"

type invertedTransferFuncType func(t core.Tensor, to core.Device) (core.Tensor, error)

var transferFunc invertedTransferFuncType

func RegisterTransferFunc(fn invertedTransferFuncType) {
	if transferFunc != nil {
		panic("gradtrack init: transferFunc has already been registered")
	}

	transferFunc = fn
}
