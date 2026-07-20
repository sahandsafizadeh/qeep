//go:build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR}/lib -Wl,-rpath=${SRCDIR}/lib -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import (
	"unsafe"

	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

func constTensor(dims []int, value float64) *CUDATensor {
	value_c := (C.double)(value)
	view_o_c := toCUDAView_C(dims)

	data_c := C.Full(value_c, view_o_c)

	return newCUDATensor(dims, data_c)
}

func eyeMatrix(d int) *CUDATensor {
	dims := []int{d, d}

	view_o_c := toCUDAView_C(dims)

	data_c := C.Eye(view_o_c)

	return newCUDATensor(dims, data_c)
}

func uniformRandomTensor(dims []int, l, u float64) *CUDATensor {
	l_c := (C.double)(l)
	u_c := (C.double)(u)
	view_o_c := toCUDAView_C(dims)

	data_c := C.RandU(l_c, u_c, view_o_c)

	return newCUDATensor(dims, data_c)
}

func normalRandomTensor(dims []int, u, s float64) *CUDATensor {
	u_c := (C.double)(u)
	s_c := (C.double)(s)
	view_o_c := toCUDAView_C(dims)

	data_c := C.RandN(u_c, s_c, view_o_c)

	return newCUDATensor(dims, data_c)
}

func tensorFromData(data any) *CUDATensor {
	var dims []int
	var inputData []C.double

	switch v := data.(type) {
	case float64:
		dims = []int{}
		inputData = append(inputData, (C.double)(v))

	case []float64:
		dims = []int{len(v)}
		for _, v0 := range v {
			inputData = append(inputData, (C.double)(v0))
		}

	case [][]float64:
		dims = []int{len(v), len(v[0])}
		for _, v0 := range v {
			for _, v1 := range v0 {
				inputData = append(inputData, (C.double)(v1))
			}
		}

	case [][][]float64:
		dims = []int{len(v), len(v[0]), len(v[0][0])}
		for _, v0 := range v {
			for _, v1 := range v0 {
				for _, v2 := range v1 {
					inputData = append(inputData, (C.double)(v2))
				}
			}
		}

	case [][][][]float64:
		dims = []int{len(v), len(v[0]), len(v[0][0]), len(v[0][0][0])}
		for _, v0 := range v {
			for _, v1 := range v0 {
				for _, v2 := range v1 {
					for _, v3 := range v2 {
						inputData = append(inputData, (C.double)(v3))
					}
				}
			}
		}

	default:
		panic("invalid input data type: data must have been based on float64 slices")
	}

	dataptr := unsafe.Pointer(&inputData[0])

	input_data_c := (*C.double)(dataptr)
	view_o_c := toCUDAView_C(dims)

	data_c := C.Of(input_data_c, view_o_c)

	return newCUDATensor(dims, data_c)
}

func tensorFromConcat(ts []*CUDATensor, dim int) *CUDATensor {
	dims := util.ConcatDims(ts, dim)
	size := len(ts)

	tsrcs := make([]C.CUDATensor, size)
	for i, t := range ts {
		tsrcs[i] = toCUDATensor_C(t)
	}

	tsptr := unsafe.Pointer(&tsrcs[0])

	ts_c := (*C.CUDATensor)(tsptr)
	size_c := (C.int)(size)
	dim_c := (C.int)(dim)
	view_o_c := toCUDAView_C(dims)

	data_c := C.Concat(ts_c, size_c, dim_c, view_o_c)

	return newCUDATensor(dims, data_c)
}
