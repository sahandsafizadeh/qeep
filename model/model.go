package model

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/model/contract"
	"github.com/sahandsafizadeh/qeep/model/internal/logger"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
	"github.com/sahandsafizadeh/qeep/model/stream"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func NewModel(
	input *stream.Stream,
	output *stream.Stream,
	lossFunc contract.Loss,
	optimizer contract.Optimizer,
) (m *Model, err error) {
	return NewModels([]*stream.Stream{input}, output, lossFunc, optimizer)
}

func NewModels(
	xs []*stream.Stream,
	y *stream.Stream,
	lossFunc contract.Loss,
	optimizer contract.Optimizer,
) (m *Model, err error) {

	if len(xs) == 0 {
		err = fmt.Errorf("")
		return
	}

	ins := make([]*node.Node, 0, len(xs))
	for _, s := range xs {
		if s == nil {
			err = fmt.Errorf("")
			return
		}

		n := s.Cursor()
		if n == nil {
			err = fmt.Errorf("")
			return
		}

		_, ok := n.Layer().(*layers.Input)
		if !ok {
			err = fmt.Errorf("")
			return
		}

		ins = append(ins, n)
	}

	if y == nil {
		err = fmt.Errorf("")
		return
	}

	// validate not null output node?

	return &Model{
		output:    y.Cursor(),
		inputs:    ins,
		loss:      lossFunc,
		optimizer: optimizer,
	}, nil
}

func (m *Model) Predict(xs []tensor.Tensor) (yp tensor.Tensor, err error) {
	err = m.disableGrad()
	if err != nil {
		return
	}

	return m.feed(xs)
}

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

	value := 0.
	result = make(map[string]float64)

	for key, metric := range metrics {
		value, err = metric.Result()
		if err != nil {
			return
		}

		result[key] = value
	}

	return result, nil
}

func (m *Model) Fit(batchGen contract.BatchGenerator, conf *FitConfig) (err error) {
	err = validateFitConfig(conf)
	if err != nil {
		return
	}

	epochLogger, err := logger.NewEpochLogger(conf.Epochs, batchGen.Count())
	if err != nil {
		return
	}

	for range conf.Epochs {
		epochLogger.StartNextEpoch()

		var xs []tensor.Tensor
		var yt tensor.Tensor
		var loss tensor.Tensor
		var epochLoss tensor.Tensor

		for batchGen.Reset(); batchGen.HasNext(); {
			xs, yt, err = batchGen.NextBatch()
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

		epochLogger.FinishEpoch(epochLoss)
	}

	return nil
}

func (m *Model) seed(xs []tensor.Tensor) (err error) {
	for i, n := range m.inputs {
		inputf := n.Layer().(*layers.Input)
		inputf.SeedFunc = func() tensor.Tensor { return xs[i] }
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

func validateFitConfig(conf *FitConfig) (err error) {
	if conf == nil {
		err = fmt.Errorf("expected fit configuration not to be nil")
		return
	}

	if conf.Epochs <= 0 {
		err = fmt.Errorf("expected the number of epochs in fit configuration to be positive: (%d) <= 0", conf.Epochs)
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
