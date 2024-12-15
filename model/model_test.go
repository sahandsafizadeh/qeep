package model_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/component/optimizers"
	"github.com/sahandsafizadeh/qeep/model"
	"github.com/sahandsafizadeh/qeep/model/batchgens"
	"github.com/sahandsafizadeh/qeep/model/stream"
)

func TestModel(t *testing.T) {

	/* ------------------------------ */

	result, err := run()
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Test Result: %v", result)

	/* ------------------------------ */

}

func run() (result map[string]float64, err error) {
	trainBatchGen, testBatchGen, err := prepareData()
	if err != nil {
		return
	}

	simpleModel, err := prepareModel()
	if err != nil {
		return
	}

	err = simpleModel.Fit(trainBatchGen, &model.FitConfig{Epochs: 10})
	if err != nil {
		return
	}

	result, err = simpleModel.Eval(testBatchGen, nil)
	if err != nil {
		return
	}

	return result, nil
}

func prepareModel() (m *model.Model, err error) {
	input := stream.Input()

	x := stream.FC(&layers.FCConfig{
		Inputs:  10,
		Outputs: 4,
	})(input)
	x = stream.Tanh()(x)

	x = stream.FC(&layers.FCConfig{
		Inputs:  4,
		Outputs: 1,
	})(x)
	output := stream.Sigmoid()(x)

	/* -------------------- */

	m, err = model.NewModel(input, output, &model.ModelConfig{
		Loss:      losses.NewBCE(),
		Optimizer: optimizers.NewSGD(nil),
	})
	if err != nil {
		return
	}

	return m, nil
}

func prepareData() (trainBatchGen, testBatchGen model.BatchGenerator, err error) {
	x := [][]float64{
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
		{9., 8., 7., 6., 5., 4., 3., 2., 1., 0.},
		{9., 8., 7., 6., 5., 4., 3., 2., 1., 0.},
		{9., 8., 7., 6., 5., 4., 3., 2., 1., 0.},
		{9., 8., 7., 6., 5., 4., 3., 2., 1., 0.},
		{9., 8., 7., 6., 5., 4., 3., 2., 1., 0.},
	}
	y := [][]float64{
		{0},
		{0},
		{0},
		{0},
		{0},
		{1},
		{1},
		{1},
		{1},
		{1},
	}

	xtr := x[:8]
	ytr := y[:8]
	xte := x[8:]
	yte := y[8:]

	trainBatchGen, err = batchgens.NewSimple(xtr, ytr, &batchgens.SimpleConfig{
		BatchSize: 4,
		Shuffle:   true,
	})
	if err != nil {
		return
	}

	testBatchGen, err = batchgens.NewSimple(xte, yte, &batchgens.SimpleConfig{
		BatchSize: 4,
		Shuffle:   true,
	})
	if err != nil {
		return
	}

	return trainBatchGen, testBatchGen, nil
}
