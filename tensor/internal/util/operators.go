package util

import "github.com/sahandsafizadeh/qeep/tensor/internal/tensor"

func BroadcastForBinaryOps(t tensor.Tensor, u tensor.Tensor) (t1 tensor.Tensor, t2 tensor.Tensor, err error) {
	shape := targetBroadcastShape(t.Shape(), u.Shape())

	t1, err = t.Broadcast(shape)
	if err != nil {
		return
	}

	t2, err = u.Broadcast(shape)
	if err != nil {
		return
	}

	return t1, t2, nil
}

func targetBroadcastShape(dims1, dims2 []int) (dims []int) {
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
	dims = make([]int, j)

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
