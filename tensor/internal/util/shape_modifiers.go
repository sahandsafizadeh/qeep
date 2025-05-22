package util

func TransposeDims(dims []int) (res []int) {
	res = make([]int, len(dims))
	copy(res, dims)

	i := len(res)
	res[i-2], res[i-1] = res[i-1], res[i-2]

	return res
}

func UnSqueezeDims(dim int, dims []int) (res []int) {
	left := dims[:dim]
	res = make([]int, len(left))
	copy(res, left)
	res = append(res, 1)

	lastDim := len(dims)
	if dim < lastDim {
		right := dims[dim:]
		res = append(res, right...)
	}

	return res
}

func SqueezeDims(dim int, dims []int) (res []int) {
	left := dims[:dim]
	res = make([]int, len(left))
	copy(res, left)

	lastDim := len(dims) - 1
	if dim < lastDim {
		right := dims[dim+1:]
		res = append(res, right...)
	}

	return res
}

func FlattenDims(dim int, dims []int) (res []int) {
	left := dims[:dim]
	res = make([]int, len(left))
	copy(res, left)

	nElems := 1

	for i := dim; i < len(dims); i++ {
		nElems *= dims[i]
	}

	res = append(res, nElems)

	return res
}

func DotDims(idims []int) (dims []int) {
	td := len(idims)
	cd := idims[:td-1]
	dims = make([]int, len(cd))
	copy(dims, cd)

	return dims
}

func MatMulDims(dims1, dims2 []int) (dims []int) {
	td := len(dims1)
	cd := dims1[:td-2]
	dims = make([]int, len(cd))
	copy(dims, cd)

	m := dims1[td-2]
	k := dims2[td-1]
	dims = append(dims, m, k)

	return dims
}
