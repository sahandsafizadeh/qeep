//go:build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR}/lib -Wl,-rpath=${SRCDIR}/lib -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import "github.com/sahandsafizadeh/qeep/tensor/internal/util"

func (t *CUDATensor) sum() float64 {
	return applyReduction(t, func(x C.CUDATensor) C.double {
		return C.Sum(x)
	})
}

func (t *CUDATensor) max() float64 {
	return applyReduction(t, func(x C.CUDATensor) C.double {
		return C.Max(x)
	})
}

func (t *CUDATensor) min() float64 {
	return applyReduction(t, func(x C.CUDATensor) C.double {
		return C.Min(x)
	})
}

func (t *CUDATensor) avg() float64 {
	return applyReduction(t, func(x C.CUDATensor) C.double {
		return C.Avg(x)
	})
}

func (t *CUDATensor) _var() float64 {
	return applyReduction(t, func(x C.CUDATensor) C.double {
		return C.Var(x)
	})
}

func (t *CUDATensor) std() float64 {
	return applyReduction(t, func(x C.CUDATensor) C.double {
		return C.Std(x)
	})
}

func (t *CUDATensor) mean() float64 {
	return t.avg()
}

func (t *CUDATensor) argmax(dim int) *CUDATensor {
	return applyDimReduction(t, dim, func(x C.CUDATensor, dim C.int, view_o C.CUDAView) *C.double {
		return C.Argmax(x, dim, view_o)
	})
}

func (t *CUDATensor) argmin(dim int) *CUDATensor {
	return applyDimReduction(t, dim, func(x C.CUDATensor, dim C.int, view_o C.CUDAView) *C.double {
		return C.Argmin(x, dim, view_o)
	})
}

func (t *CUDATensor) sumAlong(dim int) *CUDATensor {
	return applyDimReduction(t, dim, func(x C.CUDATensor, dim C.int, view_o C.CUDAView) *C.double {
		return C.SumAlong(x, dim, view_o)
	})
}

func (t *CUDATensor) maxAlong(dim int) *CUDATensor {
	return applyDimReduction(t, dim, func(x C.CUDATensor, dim C.int, view_o C.CUDAView) *C.double {
		return C.MaxAlong(x, dim, view_o)
	})
}

func (t *CUDATensor) minAlong(dim int) *CUDATensor {
	return applyDimReduction(t, dim, func(x C.CUDATensor, dim C.int, view_o C.CUDAView) *C.double {
		return C.MinAlong(x, dim, view_o)
	})
}

func (t *CUDATensor) avgAlong(dim int) *CUDATensor {
	return applyDimReduction(t, dim, func(x C.CUDATensor, dim C.int, view_o C.CUDAView) *C.double {
		return C.AvgAlong(x, dim, view_o)
	})
}

func (t *CUDATensor) varAlong(dim int) *CUDATensor {
	return applyDimReduction(t, dim, func(x C.CUDATensor, dim C.int, view_o C.CUDAView) *C.double {
		return C.VarAlong(x, dim, view_o)
	})
}

func (t *CUDATensor) stdAlong(dim int) *CUDATensor {
	return applyDimReduction(t, dim, func(x C.CUDATensor, dim C.int, view_o C.CUDAView) *C.double {
		return C.StdAlong(x, dim, view_o)
	})
}

func (t *CUDATensor) meanAlong(dim int) *CUDATensor {
	return t.avgAlong(dim)
}

func applyReduction(x *CUDATensor, rf_c reducerFunc_C) float64 {
	x_c := toCUDATensor_C(x)

	data_c := rf_c(x_c)

	return float64(data_c)
}

func applyDimReduction(x *CUDATensor, dim int, drf_c dimReducerFunc_C) *CUDATensor {
	dims := util.SqueezeDims(dim, x.dims)

	x_c := toCUDATensor_C(x)
	dim_c := (C.int)(dim)
	view_o_c := toCUDAView_C(dims)

	data_c := drf_c(x_c, dim_c, view_o_c)

	return newCUDATensor(dims, data_c)
}
