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
		lnpos += t.strd[i] * idx
	}

	return t.data[lnpos]
}

func (t *CPUTensor) slice(index []tensor.Range) *CPUTensor {
	cidx := util.CompleteIndex(index, t.dims)
	dims := util.IndexToDims(cidx)
	tidx := make([]int, len(dims))

	for i, r := range cidx {
		tidx[i] = r.From
	}

	return newTensorWithElementWiseInit(dims, func() float64 {
		defer func() {
			for i := len(tidx) - 1; i >= 0; {
				if tidx[i] < cidx[i].To-1 {
					tidx[i]++
					break
				} else {
					tidx[i] = cidx[i].From
					i--
				}
			}
		}()

		return t.at(tidx)
	})
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
