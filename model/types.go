package model

import (
	"github.com/sahandsafizadeh/qeep/model/internal/contract"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
)

// Model is a neural network built from input and output streams, a loss, and an optimizer.
// It supports training via Fit, inference via Predict, and evaluation via Eval.
type Model struct {
	inputs    []*node.Node
	output    *node.Node
	loss      contract.Loss
	optimizer contract.Optimizer
}

// ModelConfig holds the loss and optimizer used to construct a Model.
type ModelConfig struct {
	Loss      contract.Loss
	Optimizer contract.Optimizer
}

// FitConfig configures a training run: number of epochs and optional validation metrics.
type FitConfig struct {
	Epochs  int
	Metrics map[string]contract.Metric
}

// Layer is a component that computes outputs from inputs in the forward pass.
type Layer = contract.Layer

// WeightedLayer is a Layer that exposes trainable weights for the optimizer.
type WeightedLayer = contract.WeightedLayer

// Loss computes a scalar loss from predicted and target tensors for backpropagation.
type Loss = contract.Loss

// Metric accumulates predictions and targets across batches.
// Call Result() after all batches to get the final scalar value (e.g. accuracy).
type Metric = contract.Metric

// Optimizer updates layer weights using gradients computed during backpropagation.
type Optimizer = contract.Optimizer

// BatchGenerator yields batches of (inputs, target) for training or evaluation.
type BatchGenerator = contract.BatchGenerator
