//go:build cuda
// +build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR} -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import (
	"unsafe"

	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

func constTensor(dims []int, value float64) (t *CUDATensor) {
	n := util.DimsToNumElems(dims)

	n_c := (C.size_t)(n)
	value_c := (C.double)(value)

	data_c := C.Full(n_c, value_c)

	return newCUDATensor(dims, data_c)
}

func eyeMatrix(d int) (t *CUDATensor) {
	dims := []int{d, d}
	n := util.DimsToNumElems(dims)

	n_c := (C.size_t)(n)
	d_c := (C.size_t)(d)

	data_c := C.Eye(n_c, d_c)

	return newCUDATensor(dims, data_c)
}

func uniformRandomTensor(dims []int, l, u float64) (t *CUDATensor) {
	n := util.DimsToNumElems(dims)

	n_c := (C.size_t)(n)
	l_c := (C.double)(l)
	u_c := (C.double)(u)

	data_c := C.RandU(n_c, l_c, u_c)

	return newCUDATensor(dims, data_c)
}

func normalRandomTensor(dims []int, u, s float64) (t *CUDATensor) {
	n := util.DimsToNumElems(dims)

	n_c := (C.size_t)(n)
	u_c := (C.double)(u)
	s_c := (C.double)(s)

	data_c := C.RandN(n_c, u_c, s_c)

	return newCUDATensor(dims, data_c)
}

func tensorFromData(data any) (t *CUDATensor) {
	var dims []int
	var inputData []C.double

	switch v := data.(type) {
	case float64:
		dims = []int{}
		inputData = append(inputData, (C.double)(v))

	case []float64:
		d0 := len(v)
		dims = []int{d0}
		for _, v0 := range v {
			inputData = append(inputData, (C.double)(v0))
		}

	case [][]float64:
		d0 := len(v)
		d1 := len(v[0])
		dims = []int{d0, d1}
		for _, v0 := range v {
			for _, v1 := range v0 {
				inputData = append(inputData, (C.double)(v1))
			}
		}

	case [][][]float64:
		d0 := len(v)
		d1 := len(v[0])
		d2 := len(v[0][0])
		dims = []int{d0, d1, d2}
		for _, v0 := range v {
			for _, v1 := range v0 {
				for _, v2 := range v1 {
					inputData = append(inputData, (C.double)(v2))
				}
			}
		}

	case [][][][]float64:
		d0 := len(v)
		d1 := len(v[0])
		d2 := len(v[0][0])
		d3 := len(v[0][0][0])
		dims = []int{d0, d1, d2, d3}
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

	n := util.DimsToNumElems(dims)
	dataptr := unsafe.Pointer(&inputData[0])

	n_c := (C.size_t)(n)
	input_data_c := (*C.double)(dataptr)

	data_c := C.Of(n_c, input_data_c)

	return newCUDATensor(dims, data_c)
}

func tensorFromConcat(ts []*CUDATensor, dim int) (o *CUDATensor) {
	dims := util.ConcatDims(ts, dim)

	size := len(ts)
	tsSrcs := make([]C.CudaData, size)
	tsDims := make([]C.DimArr, size)

	for i, t := range ts {
		tsSrcs[i] = getCudaDataOf(t)
		tsDims[i] = getDimArrOf(t.dims)
	}

	srcsptr := unsafe.Pointer(&tsSrcs[0])
	dimsptr := unsafe.Pointer(&tsDims[0])

	srcs_c := (*C.CudaData)(srcsptr)
	dims_src_c := (*C.DimArr)(dimsptr)
	size_c := (C.size_t)(size)
	dim_c := (C.int)(dim)
	dims_dst_c := getDimArrOf(dims)

	data_c := C.Concat(srcs_c, dims_src_c, size_c, dim_c, dims_dst_c)

	return newCUDATensor(dims, data_c)
}
