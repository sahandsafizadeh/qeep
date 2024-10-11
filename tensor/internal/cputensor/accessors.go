package cputensor

import "github.com/sahandsafizadeh/qeep/tensor"

func (t *CPUTensor) numElems() (n int) {
	n = 1
	for _, dim := range t.dims {
		n *= dim
	}

	return n
}

func (t *CPUTensor) dataAt(index []int) (data any) {
	data = t.data
	for _, i := range index {
		data = data.([]any)[i]
	}

	return data
}

func (t *CPUTensor) slice(index []tensor.Range) (o *CPUTensor) {
	return t.copiedSliceOf(completeIndex(index, t.dims))
}

func (t *CPUTensor) patch(index []tensor.Range, u *CPUTensor) (o *CPUTensor) {
	return t.copiedWithPatchOf(completeIndex(index, u.dims), u)
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

	dims := make([]int, len(index))
	for i, idx := range index {
		dims[i] = idx.To - idx.From
	}

	o = new(CPUTensor)
	o.dims = dims
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

func completeIndex(index []tensor.Range, dims []int) (cidx []tensor.Range) {
	cidx = make([]tensor.Range, len(dims))
	for i := range cidx {
		// special case of all elements along dim
		if i >= len(index) || (index[i].From == 0 && index[i].To == 0) {
			cidx[i] = tensor.Range{From: 0, To: dims[i]}
		} else {
			cidx[i] = index[i]
		}
	}

	return cidx
}
