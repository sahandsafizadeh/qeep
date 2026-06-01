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
func NewModel(input *stream.Stream, output *stream.Stream, conf *ModelConfig) (m *Model, err error) {
	m, err = newModel(input, output, conf)
	if err != nil {
		return m, fmt.Errorf("NewModel failed: %w", err)
	}

	return m, nil
}

func newModel(input *stream.Stream, output *stream.Stream, conf *ModelConfig) (m *Model, err error) {
	return NewMultiInputModel([]*stream.Stream{input}, output, conf)
}

// NewMultiInputModel builds a Model with one or more input streams, the given output stream, and config.
func NewMultiInputModel(inputs []*stream.Stream, output *stream.Stream, conf *ModelConfig) (m *Model, err error) {
	m, err = newMultiInputModel(inputs, output, conf)
	if err != nil {
		return m, fmt.Errorf("NewMultiInputModel failed: %w", err)
	}

	return m, nil
}

func newMultiInputModel(inputs []*stream.Stream, output *stream.Stream, conf *ModelConfig) (m *Model, err error) {
	err = validateModelConfig(conf)
	if err != nil {
		return m, err
	}

	err = validateModelStreams(inputs, output)
	if err != nil {
		return m, err
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
func (m *Model) Predict(xs []tensor.Tensor) (yp tensor.Tensor, err error) {
	yp, err = m.predict(xs)
	if err != nil {
		return yp, fmt.Errorf("Predict operation failed: %w", err)
	}

	return yp, nil
}

func (m *Model) predict(xs []tensor.Tensor) (yp tensor.Tensor, err error) {
	err = m.disableGrad()
	if err != nil {
		return yp, err
	}

	return m.feed(xs)
}

// Eval runs Predict over all batches from batchGen and aggregates the given metrics.
func (m *Model) Eval(batchGen contract.BatchGenerator, metrics map[string]contract.Metric) (result map[string]float64, err error) {
	result, err = m.eval(batchGen, metrics)
	if err != nil {
		return result, fmt.Errorf("Eval operation failed: %w", err)
	}

	return result, nil
}

func (m *Model) eval(batchGen contract.BatchGenerator, metrics map[string]contract.Metric) (result map[string]float64, err error) {
	for batchGen.Reset(); batchGen.HasNext(); {
		xs, yt, err := batchGen.NextBatch()
		if err != nil {
			return result, err
		}

		yp, err := m.Predict(xs)
		if err != nil {
			return result, err
		}

		for _, metric := range metrics {
			err = metric.Accumulate(yp, yt)
			if err != nil {
				return result, err
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
func (m *Model) Fit(
	trainBatchGen contract.BatchGenerator,
	validBatchGen contract.BatchGenerator,
	conf *FitConfig,
) (err error) {
	err = m.validateFitConfig(conf)
	if err != nil {
		return fmt.Errorf("Fit config data validation failed: %w", err)
	}

	err = m.fit(trainBatchGen, validBatchGen, conf)
	if err != nil {
		return fmt.Errorf("Fit operation failed: %w", err)
	}

	return nil
}

func (m *Model) fit(
	trainBatchGen contract.BatchGenerator,
	validBatchGen contract.BatchGenerator,
	conf *FitConfig,
) (err error) {
	epochLogger, err := logger.NewEpochLogger(conf.Epochs, trainBatchGen.Count())
	if err != nil {
		return err
	}

	for range conf.Epochs {
		epochLogger.StartNextEpoch()

		var epochLoss tensor.Tensor
		var validResult map[string]float64

		for trainBatchGen.Reset(); trainBatchGen.HasNext(); {
			xs, yt, err := trainBatchGen.NextBatch()
			if err != nil {
				return err
			}

			loss, err := m.trainStep(xs, yt)
			if err != nil {
				return err
			}

			epochLoss, err = accumulateLoss(loss, epochLoss)
			if err != nil {
				return err
			}

			epochLogger.ProgressBatch()
		}

		if validBatchGen != nil {
			validResult, err = m.Eval(validBatchGen, conf.Metrics)
			if err != nil {
				return err
			}
		}

		epochLogger.FinishEpoch(epochLoss, validResult)
	}

	return nil
}

func (m *Model) seed(xs []tensor.Tensor) (err error) {
	err = m.validateSeedInputs(xs)
	if err != nil {
		return err
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
		return yp, err
	}

	err = m.forward()
	if err != nil {
		return yp, err
	}

	return m.output.Result(), nil
}

func (m *Model) trainStep(xs []tensor.Tensor, yt tensor.Tensor) (loss tensor.Tensor, err error) {
	err = m.enableGrad()
	if err != nil {
		return loss, err
	}

	yp, err := m.feed(xs)
	if err != nil {
		return loss, err
	}

	loss, err = m.loss.Compute(yp, yt)
	if err != nil {
		return loss, err
	}

	err = tensor.BackPropagate(loss)
	if err != nil {
		return loss, err
	}

	err = m.optimize()
	if err != nil {
		return loss, err
	}

	return loss, nil
}

/* ----- helpers ----- */

func (m *Model) validateSeedInputs(xs []tensor.Tensor) (err error) {
	if len(xs) != len(m.inputs) {
		return fmt.Errorf("expected exactly (%d) input tensors: got (%d)", len(m.inputs), len(xs))
	}

	return nil
}

func (m *Model) validateFitConfig(conf *FitConfig) (err error) {
	if conf == nil {
		return fmt.Errorf("expected config not to be nil")
	}

	if conf.Epochs <= 0 {
		return fmt.Errorf("expected 'Epochs' to be positive: got (%d)", conf.Epochs)
	}

	return nil
}

func validateModelConfig(conf *ModelConfig) (err error) {
	if conf == nil {
		return fmt.Errorf("expected config not to be nil")
	}

	return nil
}

func validateModelStreams(inputs []*stream.Stream, output *stream.Stream) (err error) {
	if len(inputs) == 0 {
		return fmt.Errorf("expected to have at least one input stream")
	}

	for i, input := range inputs {
		if input == nil {
			return fmt.Errorf("expected input stream at position (%d) not to be nil", i)
		}

		n, ok := input.Cursor().(*node.Node)
		if !ok || n == nil {
			return fmt.Errorf("expected input stream at position (%d) to be proparely initialized", i)
		}

		_, ok = n.Layer().(*layers.Input)
		if !ok {
			return fmt.Errorf("expected input stream at position (%d) to contain layer of type 'Input'", i)
		}
	}

	if output == nil {
		return fmt.Errorf("expected output stream not to be nil")
	}

	n, ok := output.Cursor().(*node.Node)
	if !ok || n == nil {
		return fmt.Errorf("expected output stream to be proparely initialized")
	}

	err = output.Error()
	if err != nil {
		return err
	}

	return nil
}

func accumulateLoss(loss tensor.Tensor, netLoss tensor.Tensor) (result tensor.Tensor, err error) {
	if netLoss == nil {
		return loss, nil
	} else {
		return loss.Add(netLoss)
	}
}
