package cputensor

func (t *CPUTensor) transpose() (o *CPUTensor) {
	elemGen := t.transposeElemGenerator()
	dims := transposeDims(t.dims)

	o = new(CPUTensor)
	o.dims = dims
	o.initWith(elemGen)

	return o
}

func (t *CPUTensor) reshape(shape []int32) (o *CPUTensor) {
	elemGen := t.linearElemGenerator()
	dims := make([]int32, len(shape))
	copy(dims, shape)

	o = new(CPUTensor)
	o.dims = dims
	o.initWith(elemGen)

	return o
}

func (t *CPUTensor) broadcast(shape []int32) (o *CPUTensor) {
	elemGen := t.broadcastElemGenerator(shape)
	dims := make([]int32, len(shape))
	copy(dims, shape)

	o = new(CPUTensor)
	o.dims = dims
	o.initWith(elemGen)

	return o
}

func (t *CPUTensor) unSqueeze(dim int32) (o *CPUTensor) {
	return t.reshape(unsqueezeDims(dim, t.dims))
}

func (t *CPUTensor) squeeze(dim int32) (o *CPUTensor) {
	return t.reshape(squeezeDims(dim, t.dims))
}

func (t *CPUTensor) flatten(fromDim int32) (o *CPUTensor) {
	return t.reshape(flattenDims(fromDim, t.dims))
}

/* ----- helpers ----- */

func (t *CPUTensor) linearElemGenerator() initializerFunc {
	state := make([]int32, len(t.dims))

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
	state := make([]int32, len(t.dims))

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

func (t *CPUTensor) broadcastElemGenerator(shape []int32) initializerFunc {
	state := make([]int32, len(t.dims))
	repeat := make([]int32, len(shape))

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

func transposeDims(dims []int32) (res []int32) {
	res = make([]int32, len(dims))
	copy(res, dims)

	i := len(res)
	res[i-2], res[i-1] = res[i-1], res[i-2]

	return res
}

func unsqueezeDims(dim int32, dims []int32) (res []int32) {
	left := dims[:dim]
	res = make([]int32, len(left))
	copy(res, left)
	res = append(res, 1)

	lastDim := int32(len(dims))
	if dim < lastDim {
		right := dims[dim:]
		res = append(res, right...)
	}

	return res
}

func squeezeDims(dim int32, dims []int32) (res []int32) {
	left := dims[:dim]
	res = make([]int32, len(left))
	copy(res, left)

	lastDim := int32(len(dims)) - 1
	if dim < lastDim {
		right := dims[dim+1:]
		res = append(res, right...)
	}

	return res
}

func flattenDims(dim int32, dims []int32) (res []int32) {
	left := dims[:dim]
	res = make([]int32, len(left))
	copy(res, left)

	// can't use int64; potential error for very large tensors
	nElems := int32(1)

	for i := dim; i < int32(len(dims)); i++ {
		nElems *= dims[i]
	}

	res = append(res, nElems)

	return res
}
