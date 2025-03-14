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
)

const (
	// https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
	dataFileAddress = "data.csv"
	validDataRatio  = 0.05
	testDataRatio   = 0.2
)

const (
	batchSize = 128
	epochs    = 25
)

func main() {
	result, err := run()
	if err != nil {
		panic(err)
	}

	for m, r := range result {
		fmt.Printf("%s: %.2f\n", m, r)
	}

	// Best Accuracy: 0.92
}

func run() (result map[string]float64, err error) {
	trainBatchGen, validBatchGen, testBatchGen, err := prepareData()
	if err != nil {
		return
	}

	bcmodel, err := prepareModel()
	if err != nil {
		return
	}

	err = bcmodel.Fit(trainBatchGen, validBatchGen, &model.FitConfig{
		Epochs: epochs,
		Metrics: map[string]model.Metric{
			"Accuracy": metrics.NewAccuracy(nil),
		},
	})
	if err != nil {
		return
	}

	result, err = bcmodel.Eval(testBatchGen, map[string]model.Metric{
		"Accuracy": metrics.NewAccuracy(nil),
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
		Inputs:  30,
		Outputs: 16,
	})(input)
	x = stream.Tanh()(x)

	x = stream.FC(&layers.FCConfig{
		Inputs:  16,
		Outputs: 1,
	})(x)
	output := stream.Sigmoid()(x)

	/* -------------------- */

	loss := losses.NewBCE()

	optimizer, err := optimizers.NewSGD(&optimizers.SGDConfig{
		LearningRate: 1e-6,
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
	})
	if err != nil {
		return
	}

	validBatchGen, err = batchgens.NewSimple(data.xValid, data.yValid, &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   false,
	})
	if err != nil {
		return
	}

	testBatchGen, err = batchgens.NewSimple(data.xTest, data.yTest, &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   false,
	})
	if err != nil {
		return
	}

	return trainBatchGen, validBatchGen, testBatchGen, nil
}
