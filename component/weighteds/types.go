package modules

import (
	qc "github.com/sahandsafizadeh/qeep/component"
	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

type WeightedComponent interface {
	qc.Component
	TrainableWeights() []*qt.Tensor
	InitWeights() error
}

type WeightedBase struct {
	isInitialized bool
	config        tinit.Config
	nInputs       int32
}
