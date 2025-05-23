//go:build cuda
// +build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${QEEP_LIB_PATH} -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import (
	"math"

	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

func (t *CUDATensor) sum() (data float64) {
	return applyReduction(t, func(src C.CudaData) (res C.double) {
		return C.Sum(src)
	})
}

func (t *CUDATensor) max() (data float64) {
	return applyReduction(t, func(src C.CudaData) (res C.double) {
		return C.Max(src)
	})
}

func (t *CUDATensor) min() (data float64) {
	return applyReduction(t, func(src C.CudaData) (res C.double) {
		return C.Min(src)
	})
}

func (t *CUDATensor) _var() (data float64) {
	return applyReduction(t, func(src C.CudaData) (res C.double) {
		return C.Var(src)
	})
}

func (t *CUDATensor) avg() (data float64) {
	return t.sum() / float64(t.n)
}

func (t *CUDATensor) std() (data float64) {
	return math.Sqrt(t._var())
}

func (t *CUDATensor) mean() (data float64) {
	return t.avg()
}

func (t *CUDATensor) argmax(dim int) (o *CUDATensor) {
	return applyDimReduction(t, dim,
		func(src C.CudaData, dim C.int, dims_src C.DimArr, dims_dst C.DimArr) (dst *C.double) {
			return C.Argmax(src, dim, dims_src, dims_dst)
		})
}

func (t *CUDATensor) argmin(dim int) (o *CUDATensor) {
	return applyDimReduction(t, dim,
		func(src C.CudaData, dim C.int, dims_src C.DimArr, dims_dst C.DimArr) (dst *C.double) {
			return C.Argmin(src, dim, dims_src, dims_dst)
		})
}

func (t *CUDATensor) sumAlong(dim int) (o *CUDATensor) {
	return applyDimReduction(t, dim,
		func(src C.CudaData, dim C.int, dims_src C.DimArr, dims_dst C.DimArr) (dst *C.double) {
			return C.SumAlong(src, dim, dims_src, dims_dst)
		})
}

func (t *CUDATensor) maxAlong(dim int) (o *CUDATensor) {
	return applyDimReduction(t, dim,
		func(src C.CudaData, dim C.int, dims_src C.DimArr, dims_dst C.DimArr) (dst *C.double) {
			return C.MaxAlong(src, dim, dims_src, dims_dst)
		})
}

func (t *CUDATensor) minAlong(dim int) (o *CUDATensor) {
	return applyDimReduction(t, dim,
		func(src C.CudaData, dim C.int, dims_src C.DimArr, dims_dst C.DimArr) (dst *C.double) {
			return C.MinAlong(src, dim, dims_src, dims_dst)
		})
}

func (t *CUDATensor) avgAlong(dim int) (o *CUDATensor) {
	return applyDimReduction(t, dim,
		func(src C.CudaData, dim C.int, dims_src C.DimArr, dims_dst C.DimArr) (dst *C.double) {
			return C.AvgAlong(src, dim, dims_src, dims_dst)
		})
}

func (t *CUDATensor) varAlong(dim int) (o *CUDATensor) {
	return applyDimReduction(t, dim,
		func(src C.CudaData, dim C.int, dims_src C.DimArr, dims_dst C.DimArr) (dst *C.double) {
			return C.VarAlong(src, dim, dims_src, dims_dst)
		})
}

func (t *CUDATensor) stdAlong(dim int) (o *CUDATensor) {
	return applyDimReduction(t, dim,
		func(src C.CudaData, dim C.int, dims_src C.DimArr, dims_dst C.DimArr) (dst *C.double) {
			return C.StdAlong(src, dim, dims_src, dims_dst)
		})
}

func (t *CUDATensor) meanAlong(dim int) (o *CUDATensor) {
	return applyDimReduction(t, dim,
		func(src C.CudaData, dim C.int, dims_src C.DimArr, dims_dst C.DimArr) (dst *C.double) {
			return C.AvgAlong(src, dim, dims_src, dims_dst)
		})
}

func applyReduction(t *CUDATensor, crf cudacReducerFunc) (res float64) {
	src_c := getCudaDataOf(t)

	data_c := crf(src_c)

	return float64(data_c)
}

func applyDimReduction(t *CUDATensor, dim int, cdrf cudacDimReducerFunc) (o *CUDATensor) {
	dims := util.SqueezeDims(dim, t.dims)

	src_c := getCudaDataOf(t)
	dim_c := (C.int)(dim)
	dims_src_c := getDimArrOf(t.dims)
	dims_dst_c := getDimArrOf(dims)

	data_c := cdrf(src_c, dim_c, dims_src_c, dims_dst_c)

	return newCUDATensor(dims, data_c)
}
