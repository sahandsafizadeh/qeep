package model_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/component/optimizers"
	"github.com/sahandsafizadeh/qeep/model"
	"github.com/sahandsafizadeh/qeep/model/batchgens"
	"github.com/sahandsafizadeh/qeep/model/stream"
	"github.com/sahandsafizadeh/qeep/tensor"
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

func TestValidationModel(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(_ tensor.Device) {

		var conf *model.ModelConfig
		var input *stream.Stream
		var output *stream.Stream
		var inputs []*stream.Stream

		/* ------------------------------ */

		conf = nil
		input = nil
		output = nil
		inputs = []*stream.Stream{input}

		_, err := model.NewMultiInputModel(inputs, output, conf)
		if err == nil {
			t.Fatalf("expected error because of nil input config")
		} else if err.Error() != "Model config data validation failed: expected config not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		/* --------------- */

		conf = new(model.ModelConfig)
		output = nil
		inputs = nil

		_, err = model.NewMultiInputModel(inputs, output, conf)
		if err == nil {
			t.Fatalf("expected error because of not having at least one input stream")
		} else if err.Error() != "Model input/output stream validation failed: expected to have at least one input stream" {
			t.Fatal("unexpected error message returned")
		}

		/* --------------- */

		conf = new(model.ModelConfig)
		input = nil
		output = nil
		inputs = []*stream.Stream{input}

		_, err = model.NewMultiInputModel(inputs, output, conf)
		if err == nil {
			t.Fatalf("expected error because of receiving nil input stream")
		} else if err.Error() != "Model input/output stream validation failed: expected input stream at position (0) not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		/* --------------- */

		conf = new(model.ModelConfig)
		input = new(stream.Stream)
		output = nil
		inputs = []*stream.Stream{input}

		_, err = model.NewMultiInputModel(inputs, output, conf)
		if err == nil {
			t.Fatalf("expected error because of receiving input stream which is not proparely initialized")
		} else if err.Error() != "Model input/output stream validation failed: expected input stream at position (0) to be proparely initialized" {
			t.Fatal("unexpected error message returned")
		}

		/* --------------- */

		conf = new(model.ModelConfig)
		input = stream.Tanh()()
		output = nil
		inputs = []*stream.Stream{input}

		_, err = model.NewMultiInputModel(inputs, output, conf)
		if err == nil {
			t.Fatalf("expected error because of receiving input stream which does not contain layer of type 'Input'")
		} else if err.Error() != "Model input/output stream validation failed: expected input stream at position (0) to contain layer of type 'Input'" {
			t.Fatal("unexpected error message returned")
		}

		/* --------------- */

		conf = new(model.ModelConfig)
		input = stream.Input()
		output = nil
		inputs = []*stream.Stream{input}

		_, err = model.NewMultiInputModel(inputs, output, conf)
		if err == nil {
			t.Fatalf("expected error because of receiving nil output stream")
		} else if err.Error() != "Model input/output stream validation failed: expected output stream not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		/* --------------- */

		conf = new(model.ModelConfig)
		input = stream.Input()
		output = new(stream.Stream)
		inputs = []*stream.Stream{input}

		_, err = model.NewMultiInputModel(inputs, output, conf)
		if err == nil {
			t.Fatalf("expected error because of receiving output stream which is not proparely initialized")
		} else if err.Error() != "Model input/output stream validation failed: expected output stream to be proparely initialized" {
			t.Fatal("unexpected error message returned")
		}

		/* --------------- */

		conf = new(model.ModelConfig)
		input = stream.Input()
		output = stream.Softmax(&activations.SoftmaxConfig{
			Dim: -1,
		})(input)
		inputs = []*stream.Stream{input}

		_, err = model.NewMultiInputModel(inputs, output, conf)
		if err == nil {
			t.Fatalf("expected error because of receiving output stream which is not proparely initialized")
		} else if err.Error() != `Model input/output stream validation failed: 
(Layer 1): Softmax config data validation failed: expected 'Dim' not to be negative: got (-1)` {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		conf = new(model.ModelConfig)
		input = stream.Input()
		output = stream.Softmax(nil)(input)
		inputs = []*stream.Stream{input}

		m, err := model.NewMultiInputModel(inputs, output, conf)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		err = m.Fit(nil, nil)
		if err == nil {
			t.Fatalf("expected error because of nil input config")
		} else if err.Error() != "Fit config data validation failed: expected config not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		err = m.Fit(nil, &model.FitConfig{
			Epochs: 0,
		})
		if err == nil {
			t.Fatalf("expected error because of nil input config")
		} else if err.Error() != "Fit config data validation failed: expected 'Epochs' to be positive: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		/* --------------- */

		_, err = m.Predict([]tensor.Tensor{nil, nil})
		if err == nil {
			t.Fatalf("expected error because of the number of input tensors not being equal to model inputs")
		} else if err.Error() != "Predict input data validation failed: expected exactly (1) input tensors: got (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
