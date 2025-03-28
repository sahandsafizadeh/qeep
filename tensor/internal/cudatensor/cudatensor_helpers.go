package cudatensor

func dimsToNumElems(dims []int) (elems int) {
	elems = 1
	for _, dim := range dims {
		elems *= dim
	}

	return elems
}
