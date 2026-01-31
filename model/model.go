package model

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/model/internal/contract"
	"github.com/sahandsafizadeh/qeep/model/internal/logger"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
	"github.com/sahandsafizadeh/qeep/model/stream"
	"github.com/sahandsafizadeh/qeep/tensor"
)

// NewModel builds a Model with a single input stream, the given output stream, and config.
// It returns an error if conf or the streams are invalid.
func NewModel(input *stream.Stream, output *stream.Stream, conf *ModelConfig) (m *Model, err error) {
	return NewMultiInputModel([]*stream.Stream{input}, output, conf)
}

// NewMultiInputModel builds a Model with one or more input streams, the given output stream, and config.
// It returns an error if conf is nil or if any input/output stream is invalid or not properly connected.
func NewMultiInputModel(inputs []*stream.Stream, output *stream.Stream, conf *ModelConfig) (m *Model, err error) {
	err = validateModelConfig(conf)
	if err != nil {
		err = fmt.Errorf("Model config data validation failed: %w", err)
		return
	}

	err = validateModelStreams(inputs, output)
	if err != nil {
		err = fmt.Errorf("Model input/output stream validation failed: %w", err)
		return
	}

	inputNodes := make([]*node.Node, len(inputs))
	for i, input := range inputs {
		inputNodes[i] = input.Cursor().(*node.Node)
	}

	outputNode := output.Cursor().(*node.Node)

	return &Model{
		inputs:    inputNodes,
		output:    outputNode,
		loss:      conf.Loss,
		optimizer: conf.Optimizer,
	}, nil
}

// Predict runs a forward pass without gradient tracking and returns the model output.
// xs must match the number of input streams; each element is the input tensor for one stream.
func (m *Model) Predict(xs []tensor.Tensor) (yp tensor.Tensor, err error) {
	err = m.disableGrad()
	if err != nil {
		return
	}

	return m.feed(xs)
}

// Eval runs Predict over all batches from batchGen and aggregates the given metrics.
// It resets the batch generator and returns a map of metric name to result (e.g. "Accuracy": 0.95).
func (m *Model) Eval(batchGen contract.BatchGenerator, metrics map[string]contract.Metric) (result map[string]float64, err error) {
	var xs []tensor.Tensor
	var yt tensor.Tensor
	var yp tensor.Tensor

	for batchGen.Reset(); batchGen.HasNext(); {
		xs, yt, err = batchGen.NextBatch()
		if err != nil {
			return
		}

		yp, err = m.Predict(xs)
		if err != nil {
			return
		}

		for _, metric := range metrics {
			err = metric.Accumulate(yp, yt)
			if err != nil {
				return
			}
		}
	}

	result = make(map[string]float64)
	for key, metric := range metrics {
		result[key] = metric.Result()
	}

	return result, nil
}

// Fit trains the model for conf.Epochs using trainBatchGen, optionally evaluating on validBatchGen each epoch.
// If validBatchGen is non-nil, conf.Metrics are computed on validation data after each epoch.
func (m *Model) Fit(
	trainBatchGen contract.BatchGenerator,
	validBatchGen contract.BatchGenerator,
	conf *FitConfig,
) (err error) {
	err = m.validateFitConfig(conf)
	if err != nil {
		err = fmt.Errorf("Fit config data validation failed: %w", err)
		return
	}

	epochLogger, err := logger.NewEpochLogger(conf.Epochs, trainBatchGen.Count())
	if err != nil {
		return
	}

	for range conf.Epochs {
		epochLogger.StartNextEpoch()

		var xs []tensor.Tensor
		var yt tensor.Tensor
		var loss tensor.Tensor
		var epochLoss tensor.Tensor
		var validResult map[string]float64

		for trainBatchGen.Reset(); trainBatchGen.HasNext(); {
			xs, yt, err = trainBatchGen.NextBatch()
			if err != nil {
				return
			}

			loss, err = m.trainStep(xs, yt)
			if err != nil {
				return
			}

			epochLoss, err = accumulateLoss(loss, epochLoss)
			if err != nil {
				return
			}

			epochLogger.ProgressBatch()
		}

		if validBatchGen != nil {
			validResult, err = m.Eval(validBatchGen, conf.Metrics)
			if err != nil {
				return
			}
		}

		epochLogger.FinishEpoch(epochLoss, validResult)
	}

	return nil
}

func (m *Model) seed(xs []tensor.Tensor) (err error) {
	err = m.validateSeedInputs(xs)
	if err != nil {
		err = fmt.Errorf("Predict input data validation failed: %w", err)
		return
	}

	for i, n := range m.inputs {
		inputl := n.Layer().(*layers.Input)
		inputl.SeedFunc = func() tensor.Tensor { return xs[i] }
	}

	return nil
}

func (m *Model) feed(xs []tensor.Tensor) (yp tensor.Tensor, err error) {
	err = m.seed(xs)
	if err != nil {
		return
	}

	err = m.forward()
	if err != nil {
		return
	}

	return m.output.Result(), nil
}

func (m *Model) trainStep(xs []tensor.Tensor, yt tensor.Tensor) (loss tensor.Tensor, err error) {
	err = m.enableGrad()
	if err != nil {
		return
	}

	yp, err := m.feed(xs)
	if err != nil {
		return
	}

	loss, err = m.loss.Compute(yp, yt)
	if err != nil {
		return
	}

	err = tensor.BackPropagate(loss)
	if err != nil {
		return
	}

	err = m.optimize()
	if err != nil {
		return
	}

	return loss, nil
}

/* ----- helpers ----- */

func (m *Model) validateSeedInputs(xs []tensor.Tensor) (err error) {
	if len(xs) != len(m.inputs) {
		err = fmt.Errorf("expected exactly (%d) input tensors: got (%d)", len(m.inputs), len(xs))
		return
	}

	return nil
}

func (m *Model) validateFitConfig(conf *FitConfig) (err error) {
	if conf == nil {
		err = fmt.Errorf("expected config not to be nil")
		return
	}

	if conf.Epochs <= 0 {
		err = fmt.Errorf("expected 'Epochs' to be positive: got (%d)", conf.Epochs)
		return
	}

	return nil
}

func validateModelConfig(conf *ModelConfig) (err error) {
	if conf == nil {
		err = fmt.Errorf("expected config not to be nil")
		return
	}

	return nil
}

func validateModelStreams(inputs []*stream.Stream, output *stream.Stream) (err error) {
	if len(inputs) == 0 {
		err = fmt.Errorf("expected to have at least one input stream")
		return
	}

	for i, input := range inputs {
		if input == nil {
			err = fmt.Errorf("expected input stream at position (%d) not to be nil", i)
			return
		}

		n, ok := input.Cursor().(*node.Node)
		if !ok || n == nil {
			err = fmt.Errorf("expected input stream at position (%d) to be proparely initialized", i)
			return
		}

		_, ok = n.Layer().(*layers.Input)
		if !ok {
			err = fmt.Errorf("expected input stream at position (%d) to contain layer of type 'Input'", i)
			return
		}
	}

	if output == nil {
		err = fmt.Errorf("expected output stream not to be nil")
		return
	}

	n, ok := output.Cursor().(*node.Node)
	if !ok || n == nil {
		err = fmt.Errorf("expected output stream to be proparely initialized")
		return
	}

	err = output.Error()
	if err != nil {
		return
	}

	return nil
}

func accumulateLoss(loss tensor.Tensor, netLoss tensor.Tensor) (tensor.Tensor, error) {
	if netLoss == nil {
		return loss, nil
	} else {
		return loss.Add(netLoss)
	}
}
