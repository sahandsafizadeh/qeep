package cputensor

import (
	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

func (t *CPUTensor) numElems() (n int) {
	return util.DimsToNumElems(t.dims)
}

func (t *CPUTensor) dataAt(index []int) (data any) {
	data = t.data
	for _, i := range index {
		data = data.([]any)[i]
	}

	return data
}

func (t *CPUTensor) slice(index []tensor.Range) (o *CPUTensor) {
	return t.copiedSliceOf(util.CompleteIndex(index, t.dims))
}

func (t *CPUTensor) patch(index []tensor.Range, u *CPUTensor) (o *CPUTensor) {
	return t.copiedWithPatchOf(util.CompleteIndex(index, u.dims), u)
}

/* ----- helpers ----- */

func (t *CPUTensor) copiedSliceOf(index []tensor.Range) (o *CPUTensor) {

	var copyData func([]tensor.Range, *any, *any)
	copyData = func(index []tensor.Range, src, dst *any) {
		if len(index) == 0 {
			*dst = (*src).(float64)
			return
		}

		idx := index[0]
		srcRows := (*src).([]any)
		dstRows := make([]any, idx.To-idx.From)
		index = index[1:]

		for i := range dstRows {
			copyData(index, &srcRows[i+idx.From], &dstRows[i])
		}

		*dst = dstRows
	}

	o = new(CPUTensor)
	o.dims = util.IndexToDims(index)
	copyData(index, &t.data, &o.data)

	return o
}

func (t *CPUTensor) copiedWithPatchOf(index []tensor.Range, u *CPUTensor) (o *CPUTensor) {

	var copyData func([]tensor.Range, *any, *any)
	copyData = func(index []tensor.Range, src, dst *any) {
		if len(index) == 0 {
			*dst = (*src).(float64)
			return
		}

		idx := index[0]
		srcRows := (*src).([]any)
		dstRows := (*dst).([]any)
		index = index[1:]

		for i := range srcRows {
			copyData(index, &srcRows[i], &dstRows[i+idx.From])
		}
	}

	o = t.slice(nil)
	copyData(index, &u.data, &o.data)

	return o
}
