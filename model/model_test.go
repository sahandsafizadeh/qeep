package model_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/initializers"
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

func TestModel(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		x := [][]float64{{-2.}, {-1.}, {1.}, {2.}}
		y := [][]float64{{0.}, {0.}, {1.}, {2.}}

		xt, err := tensor.TensorOf(x, conf)
		if err != nil {
			t.Fatal(err)
		}

		batchGen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
			BatchSize: 2,
			Device:    dev,
		})
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		var (
			wInitializer = initializers.NewFull(&initializers.FullConfig{Value: -2.})
			bInitializer = initializers.NewFull(&initializers.FullConfig{Value: -1.})
		)

		input := stream.Input()
		hidden := stream.FC(&layers.FCConfig{
			Inputs:  1,
			Outputs: 1,
			Initializers: map[string]layers.Initializer{
				"Weight": wInitializer,
				"Bias":   bInitializer,
			},
		})(input)
		output := stream.Relu()(hidden)

		/* --------------- */

		var (
			loss      = losses.NewMSE()
			metric    = metrics.NewMSE()
			optimizer = optimizers.NewSGD(&optimizers.SGDConfig{LearningRate: 0.5})
		)

		m, err := model.NewModel(input, output, &model.ModelConfig{
			Loss:      loss,
			Optimizer: optimizer,
		})
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		var (
			xs           = []tensor.Tensor{xt}
			metricKey    = "MSE"
			modelMetrics = map[string]model.Metric{
				metricKey: metric,
			}
		)

		/* --------------- */

		act, err := m.Predict(xs)
		if err != nil {
			t.Fatal(err)
		}

		exp, err := tensor.TensorOf([][]float64{{3.}, {1.}, {0.}, {0.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		result, err := m.Eval(batchGen, modelMetrics)
		if err != nil {
			t.Fatal(err)
		}

		metricValue := result[metricKey]

		if !(3.74-1e-10 < metricValue && metricValue < 3.76+1e-10) {
			t.Fatalf("expected metric value to be (3.75): got (%f)", metricValue)
		}

		/* ------------------------------ */

		err = m.Fit(batchGen, &model.FitConfig{Epochs: 2})
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act, err = m.Predict(xs)
		if err != nil {
			t.Fatal(err)
		}

		exp, err = tensor.TensorOf([][]float64{{0.}, {0.}, {0.}, {0.}}, conf)
		if err != nil {
			t.Fatal(err)
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		result, err = m.Eval(batchGen, modelMetrics)
		if err != nil {
			t.Fatal(err)
		}

		metricValue = result[metricKey]

		if !(2.4-1e-10 < metricValue && metricValue < 2.6+1e-10) {
			t.Fatalf("expected metric value to be (2.5): got (%f)", metricValue)
		}

		/* ------------------------------ */

	})
}

func TestErrorHandlingModel(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(_ tensor.Device) {
		// error in feed (m.forward) in both eval and fit
		// error in loss.compute
		// error in m.optimize
		// error in metric accumulate
		// error in batchgen.nextbatch in both eval and fit
		// error in epoch logger for negative count in batchgen
	})
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
