package util

import "github.com/sahandsafizadeh/qeep/tensor/internal/tensor"

func DimsToNumElems(dims []int) int {
	elems := 1
	for _, dim := range dims {
		elems *= dim
	}

	return elems
}

func IndexToDims(index []tensor.Range) []int {
	dims := make([]int, len(index))
	for i, idx := range index {
		dims[i] = idx.To - idx.From
	}

	return dims
}

func CompleteIndex(index []tensor.Range, dims []int) []tensor.Range {
	cidx := make([]tensor.Range, len(dims))
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
