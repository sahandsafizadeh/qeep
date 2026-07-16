//go:build cuda

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

func toCUDATensor_C(t *CUDATensor) C.CUDATensor {
	ofst_c := (C.int)(t.ofst)
	strd_c := toDimArr_C(t.strd)
	dims_c := toDimArr_C(t.dims)
	arr_c := (*C.double)(t.data)
	size_c := (C.size_t)(t.numElems())

	return C.CUDATensor{
		view: C.CUDAView{
			ofst: ofst_c,
			strd: strd_c,
			dims: dims_c,
		},
		data: C.CUDAData{
			arr:  arr_c,
			size: size_c,
		},
	}
}

func toDimArr_C(dims []int) C.DimArr {
	var arr_c [C.MAX_DIMS]C.int
	for i, d := range dims {
		arr_c[i] = (C.int)(d)
	}

	size_c := (C.size_t)(len(dims))

	return C.DimArr{
		arr:  arr_c,
		size: size_c,
	}
}

func toRangeArr_C(index []tensor.Range) C.RangeArr {
	var arr_c [C.MAX_DIMS]C.Range
	for i, r := range index {
		from_c := (C.int)(r.From)
		to_c := (C.int)(r.To)

		arr_c[i] = C.Range{
			from: from_c,
			to:   to_c,
		}
	}

	size_c := (C.size_t)(len(index))

	return C.RangeArr{
		arr:  arr_c,
		size: size_c,
	}
}
