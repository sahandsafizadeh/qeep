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

type ModelConfig struct {
	Loss      contract.Loss
	Optimizer contract.Optimizer
}

type FitConfig struct {
	Epochs  int
	Metrics map[string]contract.Metric
}

type Layer = contract.Layer
type WeightedLayer = contract.WeightedLayer
type Loss = contract.Loss
type Metric = contract.Metric
type Optimizer = contract.Optimizer
type BatchGenerator = contract.BatchGenerator
