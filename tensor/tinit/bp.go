package tinit

import (
	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
)

func BackProp(t qt.Tensor) (err error) {
	err = validateTensorDevice(t)
	if err != nil {
		return
	}

	return gradtrack.BackPropagate(t)
}
