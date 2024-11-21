package tinit

import (
	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
	qt "github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
)

func BackProp(t qt.Tensor) (err error) {
	err = validateTensorDevice(t)
	if err != nil {
		return
	}

	return gradtrack.BackPropagate(t)
}
