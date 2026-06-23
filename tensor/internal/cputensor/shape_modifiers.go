package cputensor

import (
	"slices"

	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

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
	if slices.Equal(shape, t.dims) {
		o := new(CPUTensor)
		o.dims = make([]int, len(t.dims))
		copy(o.dims, t.dims)
		o.strd = make([]int, len(t.strd))
		copy(o.strd, t.strd)

		o.data = t.data // reuse data

		return o
	}

	ofst := len(shape) - len(t.dims)
	dstidx := make([]int, len(shape))
	srcidx := make([]int, len(t.dims))

	return newTensorWithElementWiseInit(shape, func() float64 {
		defer updateElementWiseIndex(dstidx, shape)

		for i := range srcidx {
			if t.dims[i] == 1 {
				srcidx[i] = 0
			} else {
				srcidx[i] = dstidx[i+ofst]
			}
		}

		return t.at(srcidx)
	})
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
