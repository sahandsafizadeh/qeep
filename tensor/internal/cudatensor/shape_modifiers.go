//go:build cuda
// +build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR} -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import "github.com/sahandsafizadeh/qeep/tensor/internal/util"

func (t *CUDATensor) transpose() (o *CUDATensor) {
	dims := util.TransposeDims(t.dims)

	src_c := getCudaDataOf(t)
	dims_src_c := getDimArrOf(t.dims)
	dims_dst_c := getDimArrOf(dims)

	data_c := C.Transpose(src_c, dims_src_c, dims_dst_c)

	return newCUDATensor(dims, data_c)
}

func (t *CUDATensor) broadcast(shape []int) (o *CUDATensor) {
	src_c := getCudaDataOf(t)
	dims_src_c := getDimArrOf(t.dims)
	dims_dst_c := getDimArrOf(shape)

	data_c := C.Broadcast(src_c, dims_src_c, dims_dst_c)

	return newCUDATensor(shape, data_c)
}

func (t *CUDATensor) reshape(shape []int) (o *CUDATensor) {
	src_c := getCudaDataOf(t)

	data_c := C.Reshape(src_c)

	return newCUDATensor(shape, data_c)
}

func (t *CUDATensor) unsqueeze(dim int) (o *CUDATensor) {
	return t.reshape(util.UnSqueezeDims(dim, t.dims))
}

func (t *CUDATensor) squeeze(dim int) (o *CUDATensor) {
	return t.reshape(util.SqueezeDims(dim, t.dims))
}

func (t *CUDATensor) flatten(fromDim int) (o *CUDATensor) {
	return t.reshape(util.FlattenDims(fromDim, t.dims))
}
