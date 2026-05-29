//go:build cuda
// +build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR}/lib -Wl,-rpath=${SRCDIR}/lib -lcudatensor
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
		return ct, fmt.Errorf("expected input tensor to be on CUDA")
	}

	return ct, nil
}

func assertCUDATensors(ts []tensor.Tensor) (cts []*CUDATensor, err error) {
	cts = make([]*CUDATensor, len(ts))
	for i, t := range ts {
		cts[i], err = assertCUDATensor(t)
		if err != nil {
			return cts, err
		}
	}

	return cts, nil
}

func getCudaDataOf(t *CUDATensor) C.CudaData {
	arr := (*C.double)(t.data)
	size := (C.size_t)(t.n)

	return C.CudaData{
		arr:  arr,
		size: size,
	}
}

func getDimArrOf(dims []int) C.DimArr {
	var arr [C.MAX_DIMS]C.int
	for i, d := range dims {
		arr[i] = (C.int)(d)
	}

	size := (C.size_t)(len(dims))

	return C.DimArr{
		arr:  arr,
		size: size,
	}
}

func getRangeArrOf(index []tensor.Range) C.RangeArr {
	var arr [C.MAX_DIMS]C.Range
	for i, r := range index {
		arr[i] = C.Range{
			from: (C.int)(r.From),
			to:   (C.int)(r.To),
		}
	}

	size := (C.size_t)(len(index))

	return C.RangeArr{
		arr:  arr,
		size: size,
	}
}
