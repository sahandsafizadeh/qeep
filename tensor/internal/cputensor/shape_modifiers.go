package cputensor

func (t *CPUTensor) transpose() (o *CPUTensor) {
	elemGen := t.transposeElemGenerator()
	dims := transposeDims(t.dims)

	o = new(CPUTensor)
	o.dims = dims
	o.initWith(elemGen)

	return o
}

func (t *CPUTensor) reshape(shape []int) (o *CPUTensor) {
	elemGen := t.linearElemGenerator()
	dims := make([]int, len(shape))
	copy(dims, shape)

	o = new(CPUTensor)
	o.dims = dims
	o.initWith(elemGen)

	return o
}

func (t *CPUTensor) broadcast(shape []int) (o *CPUTensor) {
	elemGen := t.broadcastElemGenerator(shape)
	dims := make([]int, len(shape))
	copy(dims, shape)

	o = new(CPUTensor)
	o.dims = dims
	o.initWith(elemGen)

	return o
}

func (t *CPUTensor) unSqueeze(dim int) (o *CPUTensor) {
	return t.reshape(unsqueezeDims(dim, t.dims))
}

func (t *CPUTensor) squeeze(dim int) (o *CPUTensor) {
	return t.reshape(squeezeDims(dim, t.dims))
}

func (t *CPUTensor) flatten(fromDim int) (o *CPUTensor) {
	return t.reshape(flattenDims(fromDim, t.dims))
}

/* ----- helpers ----- */

func (t *CPUTensor) linearElemGenerator() initializerFunc {
	state := make([]int, len(t.dims))

	return func() any {
		elem := t.dataAt(state)

		i := len(t.dims) - 1
		for i >= 0 {
			if state[i] < t.dims[i]-1 {
				state[i]++
				break
			} else {
				state[i] = 0
				i--
			}
		}

		return elem
	}
}

func (t *CPUTensor) transposeElemGenerator() initializerFunc {
	state := make([]int, len(t.dims))

	return func() any {
		elem := t.dataAt(state)

		i := len(t.dims) - 2
		for i >= 0 {
			if state[i] < t.dims[i]-1 {
				state[i]++
				break
			} else {
				state[i] = 0

				switch i {
				case len(t.dims) - 2:
					i += 1
				case len(t.dims) - 1:
					i -= 2
				default:
					i--
				}
			}
		}

		return elem
	}
}

func (t *CPUTensor) broadcastElemGenerator(shape []int) initializerFunc {
	state := make([]int, len(t.dims))
	repeat := make([]int, len(shape))

	return func() any {
		elem := t.dataAt(state)

		i := len(t.dims) - 1
		j := len(shape) - 1

		for j >= 0 {
			if i >= 0 && state[i] < t.dims[i]-1 {
				state[i]++
				break

			} else if i >= 0 {
				state[i] = 0
				repeat[j]++

				if t.dims[i] == shape[j] || repeat[j] == shape[j] {
					repeat[j] = 0
					i--
					j--
				} else {
					break
				}

			} else {
				repeat[j]++

				if repeat[j] == shape[j] {
					repeat[j] = 0
					j--
				} else {
					break
				}
			}
		}

		return elem
	}
}

func transposeDims(dims []int) (res []int) {
	res = make([]int, len(dims))
	copy(res, dims)

	i := len(res)
	res[i-2], res[i-1] = res[i-1], res[i-2]

	return res
}

func unsqueezeDims(dim int, dims []int) (res []int) {
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

func squeezeDims(dim int, dims []int) (res []int) {
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

func flattenDims(dim int, dims []int) (res []int) {
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
