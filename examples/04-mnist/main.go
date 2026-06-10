package main

import (
	"fmt"
	"os"
	"runtime/pprof"

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
	validDataRatio   = 0.05
)

const (
	batchSize = 100
	epochs    = 1
	dev       = tensor.CPU
)

func main() {
	f, _ := os.Create("cpu.prof")
	defer f.Close()

	// Start the profiler
	pprof.StartCPUProfile(f)
	defer pprof.StopCPUProfile()

	result, err := run()
	if err != nil {
		panic(err)
	}

	for m, r := range result {
		fmt.Printf("%s: %.2f\n", m, r)
	}
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

	x := stream.FC(&layers.FCConfig{Outputs: 500, Device: dev})(input)
	x = stream.Relu()(x)

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

	trainBatchGen, err = batchgens.NewSimple(data.xTrain[:5000], data.yTrain[:5000], &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   true,
		Device:    dev,
	})
	if err != nil {
		return trainBatchGen, validBatchGen, testBatchGen, err
	}

	validBatchGen, err = batchgens.NewSimple(data.xValid[:200], data.yValid[:200], &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   false,
		Device:    dev,
	})
	if err != nil {
		return trainBatchGen, validBatchGen, testBatchGen, err
	}

	testBatchGen, err = batchgens.NewSimple(data.xTest[:5000], data.yTest[:5000], &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   false,
		Device:    dev,
	})
	if err != nil {
		return trainBatchGen, validBatchGen, testBatchGen, err
	}

	return trainBatchGen, validBatchGen, testBatchGen, nil
}
