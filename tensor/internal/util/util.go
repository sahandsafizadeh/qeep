package util

import "github.com/sahandsafizadeh/qeep/tensor/internal/tensor"

func DimsToNumElems(dims []int) (elems int) {
	elems = 1
	for _, dim := range dims {
		elems *= dim
	}

	return elems
}

func CompleteIndex(index []tensor.Range, dims []int) (cidx []tensor.Range) {
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
