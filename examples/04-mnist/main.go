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
	/*
		Download MNIST dataset in CSV format from Kaggle: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
		unzip mnist_train.csv.zip
		unzip mnist_test.csv.zip
	*/
	trainFileAddress = "mnist_train.csv"
	testFileAddress  = "mnist_test.csv"
	validDataRatio   = 0.1
)

const (
	batchSize = 64
	epochs    = 5
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
	trainBatchGen, validBatchGen, testBatchGen, err := prepareDataBatches()
	if err != nil {
		return result, err
	}

	mnistModel, err := prepareModel()
	if err != nil {
		return result, err
	}

	err = mnistModel.Fit(trainBatchGen, validBatchGen, &model.FitConfig{
		Epochs: epochs,
		Metrics: map[string]model.Metric{
			"Accuracy": metrics.NewAccuracy(&metrics.AccuracyConfig{
				OneHotMode: true,
			}),
		},
	})
	if err != nil {
		return result, err
	}

	result, err = mnistModel.Eval(testBatchGen, map[string]model.Metric{
		"Accuracy": metrics.NewAccuracy(&metrics.AccuracyConfig{
			OneHotMode: true,
		}),
	})
	if err != nil {
		return result, err
	}

	return result, nil
}

/* ----- model preparation ----- */

func prepareModel() (m *model.Model, err error) {
	input := stream.Input()

	x := stream.FC(&layers.FCConfig{Outputs: 256, Device: dev})(input)
	x = stream.BatchNorm(&layers.BatchNormConfig{Device: dev})(x)
	x = stream.Relu()(x)
	x = stream.Dropout(&layers.DropoutConfig{Rate: 0.2})(x)

	x = stream.FC(&layers.FCConfig{Outputs: 128, Device: dev})(x)
	x = stream.BatchNorm(&layers.BatchNormConfig{Device: dev})(x)
	x = stream.Relu()(x)
	x = stream.Dropout(&layers.DropoutConfig{Rate: 0.2})(x)

	x = stream.FC(&layers.FCConfig{Outputs: 10, Device: dev})(x)
	output := stream.Softmax(&activations.SoftmaxConfig{Dim: 1})(x)

	/* -------------------- */

	loss := losses.NewCE()

	optimizer, err := optimizers.NewAdam(nil)
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

func prepareDataBatches() (trainBatchGen, validBatchGen, testBatchGen model.BatchGenerator, err error) {
	data, err := prepareData()
	if err != nil {
		return trainBatchGen, validBatchGen, testBatchGen, err
	}

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
