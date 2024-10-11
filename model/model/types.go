package model

import (
	qc "github.com/sahandsafizadeh/qeep/component"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
	qt "github.com/sahandsafizadeh/qeep/tensor"
)

type Model struct {
	output    *node.Node
	inputs    []*node.Node
	lossFunc  qc.LossFunction
	optimizer qc.Optimizer
}

type FitConfig struct {
	Epochs    int
	BatchSize int
}

type nodeApplyFunc func(*node.Node) error
type batchGeneratorFunc func() (bxs []qt.Tensor, by qt.Tensor, err error)
