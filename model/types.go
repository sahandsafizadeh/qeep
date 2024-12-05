package model

import (
	"github.com/sahandsafizadeh/qeep/model/internal/node"
	"github.com/sahandsafizadeh/qeep/model/internal/types"
)

type Model struct {
	inputs    []*node.Node
	output    *node.Node
	lossFunc  types.Loss
	optimizer types.Optimizer
}

type FitConfig struct {
	Epochs int
}

type nodeApplyFunc func(*node.Node) error
