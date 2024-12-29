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
)

const (
	// https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
	dataFileAddress = "data.csv"
	testDataRatio   = 0.2
)

const (
	batchSize = 32
	epochs    = 10
)

func main() {
	result, err := run()
	if err != nil {
		panic(err)
	}

	for m, r := range result {
		fmt.Printf("%s: %.2f\n", m, r)
	}
}

func run() (result map[string]float64, err error) {
	trainBatchGen, testBatchGen, err := prepareData()
	if err != nil {
		return
	}

	irmodel, err := prepareModel()
	if err != nil {
		return
	}

	err = irmodel.Fit(trainBatchGen, &model.FitConfig{
		Epochs: epochs,
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
		Outputs: 64,
	})(input)
	x = stream.Relu()(x)

	x = stream.FC(&layers.FCConfig{
		Inputs:  64,
		Outputs: 32,
	})(x)
	x = stream.Relu()(x)

	x = stream.FC(&layers.FCConfig{
		Inputs:  32,
		Outputs: 3,
	})(x)
	output := stream.Softmax(&activations.SoftmaxConfig{
		Dim: 1,
	})(x)

	/* -------------------- */

	m, err = model.NewModel(input, output, &model.ModelConfig{
		Loss:      losses.NewCE(),
		Optimizer: optimizers.NewSGD(nil),
	})
	if err != nil {
		return
	}

	return m, nil
}

/* ----- data preparation ----- */

func prepareData() (trainBatchGen, testBatchGen model.BatchGenerator, err error) {
	x, y, err := loadData()
	if err != nil {
		return
	}

	xtr, xte, ytr, yte := splitData(x, y)

	preprocessData(xtr, xte)

	trainBatchGen, err = batchgens.NewSimple(xtr, ytr, &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   true,
	})
	if err != nil {
		return
	}

	testBatchGen, err = batchgens.NewSimple(xte, yte, &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   true,
	})
	if err != nil {
		return
	}

	return trainBatchGen, testBatchGen, nil
}
