package tinit

import (
	"github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
)

func BackProp(t tensor.Tensor) (err error) {
	return gradtrack.BackPropagate(t)
}
