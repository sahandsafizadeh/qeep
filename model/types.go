package model

import (
	"github.com/sahandsafizadeh/qeep/model/internal/contract"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
)

type Model struct {
	inputs    []*node.Node
	output    *node.Node
	loss      contract.Loss
	optimizer contract.Optimizer
}

type FitConfig struct {
	Epochs int
}

type nodeApplyFunc func(*node.Node) error
