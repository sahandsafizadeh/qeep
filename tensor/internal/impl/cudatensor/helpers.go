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
	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
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
	ofst_c := (C.size_t)(t.ofst)
	strd_c := toDimArr_C(t.strd)
	dims_c := toDimArr_C(t.dims)
	size_c := (C.size_t)(t.numElems())
	arr_c := (*C.double)(t.sbuf.data)

	return C.CUDATensor{
		view: C.CUDAView{
			ofst: ofst_c,
			strd: strd_c,
			dims: dims_c,
		},
		data: C.CUDAData{
			size: size_c,
			arr:  arr_c,
		},
	}
}

func toCUDAView_C(dims []int) C.CUDAView {
	strd := util.DimsToStrides(dims)

	ofst_c := (C.size_t)(0)
	strd_c := toDimArr_C(strd)
	dims_c := toDimArr_C(dims)

	return C.CUDAView{
		ofst: ofst_c,
		strd: strd_c,
		dims: dims_c,
	}
}

func toDimArr_C(dims []int) C.DimArr {
	size_c := (C.int)(len(dims))

	var arr_c [C.MAX_DIMS]C.size_t
	for i, d := range dims {
		arr_c[i] = (C.size_t)(d)
	}

	return C.DimArr{
		size: size_c,
		arr:  arr_c,
	}
}

func toRangeArr_C(index []tensor.Range) C.RangeArr {
	size_c := (C.int)(len(index))

	var arr_c [C.MAX_DIMS]C.Range
	for i, r := range index {
		arr_c[i] = C.Range{
			from: (C.size_t)(r.From),
			to:   (C.size_t)(r.To),
		}
	}

	return C.RangeArr{
		size: size_c,
		arr:  arr_c,
	}
}
