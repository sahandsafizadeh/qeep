package dimsutil

import "github.com/sahandsafizadeh/qeep/tensor/internal/core"

func DimsToNumElems(dims []int) int {
	elems := 1
	for _, dim := range dims {
		elems *= dim
	}

	return elems
}

func DimsToStrides(dims []int) []int {
	lend := len(dims)
	strd := make([]int, lend)
	if lend == 0 {
		return strd
	}

	strd[lend-1] = 1
	for i := lend - 2; i >= 0; i-- {
		strd[i] = strd[i+1] * dims[i+1]
	}

	return strd
}

func IndexToDims(index []core.Range) []int {
	dims := make([]int, len(index))
	for i, idx := range index {
		dims[i] = idx.To - idx.From
	}

	return dims
}

func CompleteIndex(index []core.Range, dims []int) []core.Range {
	cidx := make([]core.Range, len(dims))
	for i := range cidx {
		// special case of all elements along dim
		if i >= len(index) || (index[i].From == 0 && index[i].To == 0) {
			cidx[i] = core.Range{From: 0, To: dims[i]}
		} else {
			cidx[i] = index[i]
		}
	}

	return cidx
}
