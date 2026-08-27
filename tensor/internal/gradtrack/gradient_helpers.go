package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor/internal/core"

func toZeros(t core.Tensor) core.Tensor {
	return t.Scale(0)
}

func toOnes(t core.Tensor) core.Tensor {
	return t.Pow(0)
}

func reducerBroadcasted(y core.Tensor, x core.Tensor, dim int) (o core.Tensor, err error) {
	o, err = y.UnSqueeze(dim)
	if err != nil {
		return o, err
	}

	o, err = o.Broadcast(x.Shape())
	if err != nil {
		return o, err
	}

	return o, nil
}
