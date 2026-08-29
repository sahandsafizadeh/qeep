package dimsutil

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/core"
)

func DotDims(idims []int) []int {
	td := len(idims)
	cd := idims[:td-1]
	dims := make([]int, len(cd))
	copy(dims, cd)

	return dims
}

func MatMulDims(dims1, dims2 []int) []int {
	td := len(dims1)
	cd := dims1[:td-2]
	dims := make([]int, len(cd))
	copy(dims, cd)

	m := dims1[td-2]
	k := dims2[td-1]
	dims = append(dims, m, k)

	return dims
}

func BroadcastForBinaryOps(t core.Tensor, u core.Tensor) (t1 core.Tensor, t2 core.Tensor, err error) {
	shape := targetBroadcastShape(t.Shape(), u.Shape())

	t1, err = t.Broadcast(shape)
	if err != nil {
		return t1, t2, fmt.Errorf("failed to broadcast first operand: %w", err)
	}

	t2, err = u.Broadcast(shape)
	if err != nil {
		return t1, t2, fmt.Errorf("failed to broadcast second operand: %w", err)
	}

	return t1, t2, nil
}

func BroadcastForMatMul(t core.Tensor, u core.Tensor) (t1 core.Tensor, t2 core.Tensor, err error) {
	dims1 := t.Shape()
	dims2 := u.Shape()
	shape := targetBroadcastShape(dims1, dims2)

	lt := len(shape)
	l1 := len(dims1)
	l2 := len(dims2)

	shape[lt-1] = dims1[l1-1]
	shape[lt-2] = dims1[l1-2]

	t1, err = t.Broadcast(shape)
	if err != nil {
		return t1, t2, fmt.Errorf("failed to broadcast first operand: %w", err)
	}

	shape[lt-1] = dims2[l2-1]
	shape[lt-2] = dims2[l2-2]

	t2, err = u.Broadcast(shape)
	if err != nil {
		return t1, t2, fmt.Errorf("failed to broadcast second operand: %w", err)
	}

	return t1, t2, nil
}

func targetBroadcastShape(dims1, dims2 []int) []int {
	var small, large []int
	if len(dims1) > len(dims2) {
		small = dims2
		large = dims1
	} else {
		small = dims1
		large = dims2
	}

	i := len(small)
	j := len(large)
	dims := make([]int, j)

	for i > 0 {
		i--
		j--
		dims[j] = max(small[i], large[j])
	}

	for j > 0 {
		j--
		dims[j] = large[j]
	}

	return dims
}
