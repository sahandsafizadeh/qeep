package main

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/component/metrics"
	"github.com/sahandsafizadeh/qeep/component/optimizers"
	"github.com/sahandsafizadeh/qeep/model"
	"github.com/sahandsafizadeh/qeep/model/batchgens"
	"github.com/sahandsafizadeh/qeep/model/stream"
	"github.com/sahandsafizadeh/qeep/tensor"
)

const (
	// https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
	dataFileAddress = "data.csv"
	validDataRatio  = 0.1
	testDataRatio   = 0.2
)

const (
	batchSize = 32
	epochs    = 1000
	dev       = tensor.CPU
)

func main() {
	result, err := run()
	if err != nil {
		panic(err)
	}

	for m, r := range result {
		fmt.Printf("%s: %.2f\n", m, r)
	}

	// Best Accuracy: 0.73
	// Total Duration (CPU: 21s, CUDA: 5m6s)
}

func run() (result map[string]float64, err error) {
	trainBatchGen, validBatchGen, testBatchGen, err := prepareData()
	if err != nil {
		return
	}

	irmodel, err := prepareModel()
	if err != nil {
		return
	}

	err = irmodel.Fit(trainBatchGen, validBatchGen, &model.FitConfig{
		Epochs: epochs,
		Metrics: map[string]model.Metric{
			"Accuracy": metrics.NewAccuracy(&metrics.AccuracyConfig{
				OneHotMode: true,
			}),
		},
	})
	if err != nil {
		return
	}

	result, err = irmodel.Eval(testBatchGen, map[string]model.Metric{
		"Accuracy": metrics.NewAccuracy(&metrics.AccuracyConfig{
			OneHotMode: true,
		}),
	})
	if err != nil {
		return
	}

	return result, nil
}

/* ----- model preparation ----- */

func prepareModel() (m *model.Model, err error) {
	input := stream.Input()

	x := stream.FC(&layers.FCConfig{
		Inputs:  4,
		Outputs: 4,
		Device:  dev,
	})(input)
	x = stream.Tanh()(x)

	x = stream.FC(&layers.FCConfig{
		Inputs:  4,
		Outputs: 3,
		Device:  dev,
	})(x)
	output := stream.Softmax(&activations.SoftmaxConfig{
		Dim: 1,
	})(x)

	/* -------------------- */

	loss := losses.NewCE()

	optimizer, err := optimizers.NewAdamW(&optimizers.AdamWConfig{
		LearningRate: 1e-5,
		WeightDecay:  optimizers.AdamWDefaultWeightDecay,
		Beta1:        optimizers.AdamWDefaultBeta1,
		Beta2:        optimizers.AdamWDefaultBeta2,
		Eps:          optimizers.AdamWDefaultEps,
	})
	if err != nil {
		return
	}

	m, err = model.NewModel(input, output, &model.ModelConfig{
		Loss:      loss,
		Optimizer: optimizer,
	})
	if err != nil {
		return
	}

	return m, nil
}

/* ----- data preparation ----- */

func prepareData() (trainBatchGen, validBatchGen, testBatchGen model.BatchGenerator, err error) {
	x, y, err := loadData()
	if err != nil {
		return
	}

	data := splitData(x, y)

	preprocessData(data)

	trainBatchGen, err = batchgens.NewSimple(data.xTrain, data.yTrain, &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   true,
		Device:    dev,
	})
	if err != nil {
		return
	}

	validBatchGen, err = batchgens.NewSimple(data.xValid, data.yValid, &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   false,
		Device:    dev,
	})
	if err != nil {
		return
	}

	testBatchGen, err = batchgens.NewSimple(data.xTest, data.yTest, &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   false,
		Device:    dev,
	})
	if err != nil {
		return
	}

	return trainBatchGen, validBatchGen, testBatchGen, nil
}
