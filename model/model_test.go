package model_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/component/metrics"
	"github.com/sahandsafizadeh/qeep/component/optimizers"
	"github.com/sahandsafizadeh/qeep/model"
	"github.com/sahandsafizadeh/qeep/model/batchgen"
	"github.com/sahandsafizadeh/qeep/model/internal/types"
	sahand "github.com/sahandsafizadeh/qeep/model/layers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestModel(t *testing.T) {

	/* -------------- data -------------- */

	trainBatchGen, testBatchGen, err := prepareData()
	if err != nil {
		return
	}

	/* ---------- architecture ---------- */

	input := sahand.Input()

	x := sahand.FC(&layers.FCConfig{Outputs: 64})(input)
	x = sahand.Tanh()(x)

	x = sahand.FC(&layers.FCConfig{Outputs: 32})(x)
	x = sahand.Sigmoid()(x)

	x = sahand.FC(&layers.FCConfig{Outputs: 16})(x)
	x = sahand.Relu()(x)

	x = sahand.FC(&layers.FCConfig{Outputs: 8})(x)
	x = sahand.LeakyRelu(nil)(x)

	x = sahand.FC(&layers.FCConfig{Outputs: 4})(x)
	output := sahand.Softmax(&activations.SoftmaxConfig{Dim: 1})(x)

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

	result, err := mod.Eval(testBatchGen, map[string]types.Metric{"Accuracy": metrics.NewAccuracy()})
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Test Result: %v", result)
}

func prepareData() (trainBatchGen, testBatchGen types.BatchGenerator, err error) {
	x, err := tensor.RandU([]int{10000, 5}, -1., 1., nil)
	if err != nil {
		return
	}

	y, err := tensor.RandN([]int{10000}, 0., 0.5, nil)
	if err != nil {
		return
	}

	xtr, err := x.Slice([]tensor.Range{{From: 0, To: 8000}})
	if err != nil {
		return
	}

	ytr, err := y.Slice([]tensor.Range{{From: 0, To: 8000}})
	if err != nil {
		return
	}

	xte, err := x.Slice([]tensor.Range{{From: 8000, To: 10000}})
	if err != nil {
		return
	}

	yte, err := y.Slice([]tensor.Range{{From: 8000, To: 10000}})
	if err != nil {
		return
	}

	trainBatchGen, err = batchgen.NewSimple(xtr, ytr, 256)
	if err != nil {
		return
	}

	testBatchGen, err = batchgen.NewSimple(xte, yte, 64)
	if err != nil {
		return
	}

	return trainBatchGen, testBatchGen, nil
}
