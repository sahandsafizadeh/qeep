package component

import qt "github.com/sahandsafizadeh/qeep/tensor"

type Component interface {
	Forward(...qt.Tensor) (qt.Tensor, error)
}
