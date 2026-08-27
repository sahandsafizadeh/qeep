package cputensor

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/core"
)

func assertCPUTensor(t core.Tensor) (ct *CPUTensor, err error) {
	ct, ok := t.(*CPUTensor)
	if !ok {
		return ct, fmt.Errorf("expected input tensor to be on CPU")
	}

	return ct, nil
}

func assertCPUTensors(ts []core.Tensor) (cts []*CPUTensor, err error) {
	cts = make([]*CPUTensor, len(ts))
	for i, t := range ts {
		cts[i], err = assertCPUTensor(t)
		if err != nil {
			return cts, err
		}
	}

	return cts, nil
}
