package tinit

import (
	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
)

func BackProp(t tensor.Tensor) (err error) {
	err = validateTensorDevice(t)
	if err != nil {
		return
	}

	return gradtrack.BackPropagate(t)
}
