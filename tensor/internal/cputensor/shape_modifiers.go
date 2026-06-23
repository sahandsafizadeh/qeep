package cputensor

import "github.com/sahandsafizadeh/qeep/tensor/internal/util"

func (t *CPUTensor) transpose() *CPUTensor {
	n := len(t.dims)

	o := new(CPUTensor)
	o.dims = util.TransposeDims(t.dims)
	o.strd = make([]int, n)
	copy(o.strd, t.strd)
	o.strd[n-2], o.strd[n-1] = o.strd[n-1], o.strd[n-2]

	o.data = t.data // reuse data

	return o
}

func (t *CPUTensor) reshape(shape []int) *CPUTensor {
	o := new(CPUTensor)
	o.dims = make([]int, len(shape))
	copy(o.dims, shape)
	o.strd = util.DimsToStrides(shape)

	o.data = t.data // reuse data

	return o
}

func (t *CPUTensor) broadcast(shape []int) *CPUTensor {
	elemGen := t.broadcastElemGenerator(shape)
	dims := make([]int, len(shape))
	copy(dims, shape)

	o := new(CPUTensor)
	o.dims = dims
	o.initWith(elemGen)

	return o
}

func (t *CPUTensor) unsqueeze(dim int) *CPUTensor {
	return t.reshape(util.UnSqueezeDims(dim, t.dims))
}

func (t *CPUTensor) squeeze(dim int) *CPUTensor {
	return t.reshape(util.SqueezeDims(dim, t.dims))
}

func (t *CPUTensor) flatten(fromDim int) *CPUTensor {
	return t.reshape(util.FlattenDims(fromDim, t.dims))
}

/* ----- helpers ----- */

func (t *CPUTensor) broadcastElemGenerator(shape []int) func() any {
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
