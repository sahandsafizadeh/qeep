package modules

import (
	qc "github.com/sahandsafizadeh/qeep/component"
	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type WeightedComponent interface {
	qc.Component
	TrainableWeights() []*qt.Tensor
}
