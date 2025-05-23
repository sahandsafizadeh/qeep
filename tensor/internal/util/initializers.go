package util

import "github.com/sahandsafizadeh/qeep/tensor/internal/tensor"

func ConcatDims[T tensor.Tensor](ts []T, dim int) (dims []int) {
	common := 0
	for _, t := range ts {
		common += t.Shape()[dim]
	}

	base := ts[0].Shape()
	dims = make([]int, len(base))
	copy(dims, base)
	dims[dim] = common

	return dims
}
