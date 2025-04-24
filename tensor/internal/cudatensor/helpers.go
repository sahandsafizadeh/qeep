package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR} -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

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

func getCudaDataOf(t *CUDATensor) (cd C.CudaData) {
	arr := (*C.double)(t.data)
	size := (C.size_t)(t.n)

	return C.CudaData{
		arr:  arr,
		size: size,
	}
}
