package cudatensor

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
)

func assertCUDATensor(t tensor.Tensor) (ct *CUDATensor, err error) {
	ct, ok := t.(*CUDATensor)
	if !ok {
		err = fmt.Errorf("expected input tensor to be on CUDA")
		return
	}

	return ct, nil
}
