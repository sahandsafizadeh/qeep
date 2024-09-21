package model

import (
	qc "github.com/sahandsafizadeh/qeep/component"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
)

type Model struct {
	output    *node.Node
	inputs    []*node.Node
	lossFunc  qc.LossFunc
	optimFunc qc.OptimizerFunc
}

type FitConfig struct {
	Epochs    int32
	BatchSize int32
}
