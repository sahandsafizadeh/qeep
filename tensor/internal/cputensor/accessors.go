package cputensor

import (
	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

func (t *CPUTensor) numElems() int {
	return util.DimsToNumElems(t.dims)
}

func (t *CPUTensor) at(index []int) float64 {
	lnpos := 0
	for i, idx := range index {
		lnpos += t.strd[i]*idx + t.ofst[i]
	}

	return t.data[lnpos]
}

func (t *CPUTensor) slice(index []tensor.Range) *CPUTensor {
	cidx := util.CompleteIndex(index, t.dims)

	o := new(CPUTensor)
	o.dims = util.IndexToDims(cidx)
	o.strd = make([]int, len(t.strd))
	copy(o.strd, t.strd)

	o.ofst = make([]int, len(t.ofst))
	for i, r := range cidx {
		o.ofst[i] = t.strd[i]*r.From + t.ofst[i]
	}

	o.data = t.data // reuse data

	return o
}

func (t *CPUTensor) patch(index []tensor.Range, u *CPUTensor) *CPUTensor {
	cidx := util.CompleteIndex(index, u.dims)
	tidx := make([]int, len(t.dims))
	uidx := make([]int, len(u.dims))

	return newTensorWithElementWiseInit(t.dims, func() float64 {
		defer updateElementWiseIndex(tidx, t.dims)

		for i, r := range cidx {
			if tidx[i] < r.From || tidx[i] >= r.To {
				return t.at(tidx)
			}
		}

		for i, r := range cidx {
			uidx[i] = tidx[i] - r.From
		}

		return u.at(uidx)
	})
}
