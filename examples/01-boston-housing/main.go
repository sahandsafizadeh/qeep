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
	// https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data
	dataFileAddress = "data.csv"
	validDataRatio  = 0.05
	testDataRatio   = 0.2
)

const (
	batchSize = 128
	epochs    = 500
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

	// Best Mean Squared Error (MSE): 45.44
	// Total Duration (CPU: 29s, CUDA: 29s)
}

func run() (result map[string]float64, err error) {
	trainBatchGen, validBatchGen, testBatchGen, err := prepareData()
	if err != nil {
		return
	}

	bhmodel, err := prepareModel()
	if err != nil {
		return
	}

	err = bhmodel.Fit(trainBatchGen, validBatchGen, &model.FitConfig{
		Epochs: epochs,
		Metrics: map[string]model.Metric{
			"MSE": metrics.NewMSE(),
		},
	})
	if err != nil {
		return
	}

	result, err = bhmodel.Eval(testBatchGen, map[string]model.Metric{
		"Mean Squared Error (MSE)": metrics.NewMSE(),
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
		Inputs:  13,
		Outputs: 32,
		Device:  dev,
	})(input)
	x = stream.Tanh()(x)

	output := stream.FC(&layers.FCConfig{
		Inputs:  32,
		Outputs: 1,
		Device:  dev,
	})(x)

	/* -------------------- */

	loss := losses.NewMSE()

	optimizer, err := optimizers.NewAdamW(nil)
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
