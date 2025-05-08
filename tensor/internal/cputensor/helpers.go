package cputensor

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
)

func assertCPUTensor(t tensor.Tensor) (ct *CPUTensor, err error) {
	ct, ok := t.(*CPUTensor)
	if !ok {
		err = fmt.Errorf("expected input tensor to be on CPU")
		return
	}

	return ct, nil
}

func assertCPUTensors(ts []tensor.Tensor) (cts []*CPUTensor, err error) {
	cts = make([]*CPUTensor, len(ts))
	for i, t := range ts {
		cts[i], err = assertCPUTensor(t)
		if err != nil {
			return
		}
	}

	return cts, nil
}
