package main

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/component/metrics"
	"github.com/sahandsafizadeh/qeep/component/optimizers"
	"github.com/sahandsafizadeh/qeep/model"
	"github.com/sahandsafizadeh/qeep/model/batchgens"
	"github.com/sahandsafizadeh/qeep/model/stream"
	"github.com/sahandsafizadeh/qeep/tensor"
)

const (
	// https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
	dataFileAddress = "data.csv"
	validDataRatio  = 0.1
	testDataRatio   = 0.2
)

const (
	batchSize = 32
	epochs    = 200
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

	// Best Accuracy: 0.98
}

func run() (result map[string]float64, err error) {
	trainBatchGen, validBatchGen, testBatchGen, err := prepareData()
	if err != nil {
		return result, err
	}

	bcmodel, err := prepareModel()
	if err != nil {
		return result, err
	}

	err = bcmodel.Fit(trainBatchGen, validBatchGen, &model.FitConfig{
		Epochs: epochs,
		Metrics: map[string]model.Metric{
			"Accuracy": metrics.NewAccuracy(nil),
		},
	})
	if err != nil {
		return result, err
	}

	result, err = bcmodel.Eval(testBatchGen, map[string]model.Metric{
		"Accuracy": metrics.NewAccuracy(nil),
	})
	if err != nil {
		return result, err
	}

	return result, nil
}

/* ----- model preparation ----- */

func prepareModel() (m *model.Model, err error) {
	input := stream.Input()

	x := stream.FC(&layers.FCConfig{Outputs: 32, Device: dev})(input)
	x = stream.BatchNorm(&layers.BatchNormConfig{Device: dev})(x)
	x = stream.Relu()(x)
	x = stream.Dropout(&layers.DropoutConfig{Rate: 0.3})(x)

	x = stream.FC(&layers.FCConfig{Outputs: 16, Device: dev})(x)
	x = stream.BatchNorm(&layers.BatchNormConfig{Device: dev})(x)
	x = stream.Relu()(x)
	x = stream.Dropout(&layers.DropoutConfig{Rate: 0.2})(x)

	x = stream.FC(&layers.FCConfig{Outputs: 1, Device: dev})(x)
	output := stream.Sigmoid()(x)

	/* -------------------- */

	loss := losses.NewBCE()

	optimizer, err := optimizers.NewAdamW(&optimizers.AdamWConfig{WeightDecay: 1e-4})
	if err != nil {
		return m, err
	}

	m, err = model.NewModel(input, output, &model.ModelConfig{
		Loss:      loss,
		Optimizer: optimizer,
	})
	if err != nil {
		return m, err
	}

	return m, nil
}

/* ----- data preparation ----- */

func prepareData() (trainBatchGen, validBatchGen, testBatchGen model.BatchGenerator, err error) {
	x, y, err := loadData()
	if err != nil {
		return trainBatchGen, validBatchGen, testBatchGen, err
	}

	data := splitData(x, y)

	preprocessData(data)

	trainBatchGen, err = batchgens.NewSimple(data.xTrain, data.yTrain, &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   true,
		Device:    dev,
	})
	if err != nil {
		return trainBatchGen, validBatchGen, testBatchGen, err
	}

	validBatchGen, err = batchgens.NewSimple(data.xValid, data.yValid, &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   false,
		Device:    dev,
	})
	if err != nil {
		return trainBatchGen, validBatchGen, testBatchGen, err
	}

	testBatchGen, err = batchgens.NewSimple(data.xTest, data.yTest, &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   false,
		Device:    dev,
	})
	if err != nil {
		return trainBatchGen, validBatchGen, testBatchGen, err
	}

	return trainBatchGen, validBatchGen, testBatchGen, nil
}
