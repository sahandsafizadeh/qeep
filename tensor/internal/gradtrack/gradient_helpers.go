package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor/internal/tensor"

func toZeros(t tensor.Tensor) (o tensor.Tensor) {
	return t.Scale(0)
}

func toOnes(t tensor.Tensor) (o tensor.Tensor) {
	return t.Pow(0)
}

func reducerBroadcasted(y tensor.Tensor, x tensor.Tensor, dim int) (o tensor.Tensor, err error) {
	o, err = y.UnSqueeze(dim)
	if err != nil {
		return
	}

	o, err = o.Broadcast(x.Shape())
	if err != nil {
		return
	}

	return o, nil
}
