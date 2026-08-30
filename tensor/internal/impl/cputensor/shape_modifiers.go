package cputensor

import (
	"slices"

	"github.com/sahandsafizadeh/qeep/tensor/internal/impl/common/dimsutil"
)

func (t *CPUTensor) transpose() *CPUTensor {
	n := len(t.dims)

	o := new(CPUTensor)
	o.ofst = t.ofst
	o.strd = make([]int, n)
	copy(o.strd, t.strd)
	o.strd[n-2], o.strd[n-1] = o.strd[n-1], o.strd[n-2]
	o.dims = dimsutil.TransposeDims(t.dims)

	o.data = t.data // reuse data

	return o
}

func (t *CPUTensor) reshape(shape []int) *CPUTensor {
	fofst := 0
	fstrd := dimsutil.DimsToStrides(t.dims)

	if t.ofst != fofst || !slices.Equal(t.strd, fstrd) { // impossible to reuse data; copy
		return t.compact(shape)
	}

	o := new(CPUTensor)
	o.ofst = 0
	o.strd = dimsutil.DimsToStrides(shape)
	o.dims = make([]int, len(shape))
	copy(o.dims, shape)

	o.data = t.data // reuse data

	return o
}

func (t *CPUTensor) flatten(fromDim int) *CPUTensor {
	return t.reshape(dimsutil.FlattenDims(fromDim, t.dims))
}

func (t *CPUTensor) unsqueeze(dim int) *CPUTensor {
	return t.reshape(dimsutil.UnSqueezeDims(dim, t.dims))
}

func (t *CPUTensor) squeeze(dim int) *CPUTensor {
	return t.reshape(dimsutil.SqueezeDims(dim, t.dims))
}

func (t *CPUTensor) broadcast(shape []int) *CPUTensor {
	o := new(CPUTensor)
	o.ofst = t.ofst
	o.strd = make([]int, len(shape))
	o.dims = make([]int, len(shape))
	copy(o.dims, shape)

	offset := len(shape) - len(t.dims)
	for i := range o.strd {
		if i < offset {
			o.strd[i] = 0
		} else if j := i - offset; t.dims[j] == 1 && shape[i] != 1 {
			o.strd[i] = 0
		} else {
			o.strd[i] = t.strd[j]
		}
	}

	o.data = t.data // reuse data

	return o
}

func (t *CPUTensor) compact(shape []int) *CPUTensor {
	index := make([]int, len(t.dims))
	return newTensorWithElementWiseInit(shape, func() float64 {
		defer updateElementWiseIndex(index, t.dims)
		return t.at(index)
	})
}
