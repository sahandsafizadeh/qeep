package cudatensor

func (t *CUDATensor) numElems() (n int) {
	return dimsToNumElems(t.dims)
}
