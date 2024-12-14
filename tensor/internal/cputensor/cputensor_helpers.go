package cputensor

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
)

func assertCPUTensor(t tensor.Tensor) (ct *CPUTensor, err error) {
	ct, ok := t.(*CPUTensor)
	if !ok {
		err = fmt.Errorf("expected input tensor to be on CPU")
		return
	}

	return ct, nil
}

func assertCPUTensors(ts []tensor.Tensor) (cts []*CPUTensor, err error) {
	cts = make([]*CPUTensor, len(ts))
	for i, t := range ts {
		cts[i], err = assertCPUTensor(t)
		if err != nil {
			return
		}
	}

	return cts, nil
}

func broadcastForBinaryOp(ct1, ct2 *CPUTensor) (bct1, bct2 *CPUTensor, err error) {
	shape := targetBroadcastDims(ct1.dims, ct2.dims)

	t1, err := ct1.Broadcast(shape)
	if err != nil {
		return
	}

	t2, err := ct2.Broadcast(shape)
	if err != nil {
		return
	}

	bct1 = t1.(*CPUTensor)
	bct2 = t2.(*CPUTensor)

	return bct1, bct2, nil
}

func broadcastForMatMul(ct1, ct2 *CPUTensor) (bct1, bct2 *CPUTensor, err error) {
	shape := targetBroadcastDims(ct1.dims, ct2.dims)

	lt := len(shape)
	l1 := len(ct1.dims)
	l2 := len(ct2.dims)

	shape[lt-1] = ct1.dims[l1-1]
	shape[lt-2] = ct1.dims[l1-2]

	t1, err := ct1.Broadcast(shape)
	if err != nil {
		return
	}

	shape[lt-1] = ct2.dims[l2-1]
	shape[lt-2] = ct2.dims[l2-2]

	t2, err := ct2.Broadcast(shape)
	if err != nil {
		return
	}

	bct1 = t1.(*CPUTensor)
	bct2 = t2.(*CPUTensor)

	return bct1, bct2, nil
}

func targetBroadcastDims(dims1, dims2 []int) (dims []int) {
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

		if small[i] > large[j] {
			dims[j] = small[i]
		} else {
			dims[j] = large[j]
		}
	}

	for j > 0 {
		j--
		dims[j] = large[j]
	}

	return dims
}
