package cputensor

import (
	"github.com/sahandsafizadeh/qeep/tensor/internal/core"
	"github.com/sahandsafizadeh/qeep/tensor/internal/impl/common/dimsutil"
)

func (t *CPUTensor) numElems() int {
	return dimsutil.DimsToNumElems(t.dims)
}

func (t *CPUTensor) at(index []int) float64 {
	lnpos := t.ofst
	for i, idx := range index {
		lnpos += t.strd[i] * idx
	}

	return t.data[lnpos]
}

func (t *CPUTensor) slice(index []core.Range) *CPUTensor {
	cidx := dimsutil.CompleteIndex(index, t.dims)

	o := new(CPUTensor)
	o.ofst = t.ofst
	o.strd = make([]int, len(t.strd))
	copy(o.strd, t.strd)
	o.dims = dimsutil.IndexToDims(cidx)

	for i, r := range cidx {
		o.ofst += t.strd[i] * r.From
	}

	o.data = t.data // reuse data

	return o
}
