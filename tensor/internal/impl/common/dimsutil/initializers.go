package dimsutil

import "github.com/sahandsafizadeh/qeep/tensor/internal/core"

func ConcatDims[T core.Tensor](ts []T, dim int) []int {
	common := 0
	for _, t := range ts {
		common += t.Shape()[dim]
	}

	base := ts[0].Shape()
	dims := make([]int, len(base))
	copy(dims, base)
	dims[dim] = common

	return dims
}
