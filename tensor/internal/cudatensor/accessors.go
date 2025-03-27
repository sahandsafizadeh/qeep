package cudatensor

func (t *CUDATensor) numElems() (n int) {
	n = 1
	for _, dim := range t.dims {
		n *= dim
	}

	return n
}
