package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR} -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import "github.com/sahandsafizadeh/qeep/tensor/internal/util"

func (t *CUDATensor) sum() (data float64) {
	return applyReduction(t, func(src *C.double, n C.size_t) (res C.double) {
		return C.Sum(src, n)
	})
}

func (t *CUDATensor) max() (data float64) {
	return applyReduction(t, func(src *C.double, n C.size_t) (res C.double) {
		return C.Max(src, n)
	})
}

func (t *CUDATensor) min() (data float64) {
	return applyReduction(t, func(src *C.double, n C.size_t) (res C.double) {
		return C.Min(src, n)
	})
}

func (t *CUDATensor) avg() (data float64) {
	return t.sum() / float64(t.n)
}

func (t *CUDATensor) mean() (data float64) {
	return t.avg()
}

func applyReduction(t *CUDATensor, crf cudacReducerFunc) (res float64) {
	dims := t.dims
	data := t.data
	n := util.DimsToNumElems(dims)

	src_c := (*C.double)(data)
	n_c := (C.size_t)(n)

	data_c := crf(src_c, n_c)

	return float64(data_c)
}
