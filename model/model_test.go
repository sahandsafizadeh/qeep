package model_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/component/metrics"
	"github.com/sahandsafizadeh/qeep/component/optimizers"
	"github.com/sahandsafizadeh/qeep/model"
	"github.com/sahandsafizadeh/qeep/model/batchgens"
	"github.com/sahandsafizadeh/qeep/model/stream"
)

func TestModel(t *testing.T) {

	/* -------------- data -------------- */

	trainBatchGen, testBatchGen, err := prepareData()
	if err != nil {
		return
	}

	/* ---------- architecture ---------- */

	input := stream.Input()

	x := stream.FC(&layers.FCConfig{Outputs: 64})(input)
	x = stream.Tanh()(x)

	x = stream.FC(&layers.FCConfig{Outputs: 32})(x)
	x = stream.Sigmoid()(x)

	x = stream.FC(&layers.FCConfig{Outputs: 16})(x)
	x = stream.Relu()(x)

	x = stream.FC(&layers.FCConfig{Outputs: 8})(x)
	x = stream.LeakyRelu(nil)(x)

	x = stream.FC(&layers.FCConfig{Outputs: 4})(x)
	output := stream.Softmax(&activations.SoftmaxConfig{Dim: 1})(x)

	/* ------------- model ------------- */

	mod, err := model.NewModel(
		input,
		output,
		losses.NewCE(),
		optimizers.NewSGD(nil),
	)
	if err != nil {
		t.Fatal(err)
	}

	/* ------------- train ------------- */

	err = mod.Fit(trainBatchGen, &model.FitConfig{Epochs: 10})
	if err != nil {
		t.Fatal(err)
	}

	/* ------------- test ------------- */

	result, err := mod.Eval(testBatchGen, map[string]model.Metric{"Accuracy": metrics.NewAccuracy()})
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Test Result: %v", result)
}

func prepareData() (trainBatchGen, testBatchGen model.BatchGenerator, err error) {
	x := [][]float64{
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
		{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.},
	}
	y := []float64{
		0,
		0,
		0,
		0,
		0,
		1,
		1,
		1,
		1,
		1,
	}

	xtr, ytr := x[:8], y[:8]
	xte, yte := x[8:], y[8:]

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
