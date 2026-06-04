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

		// ============================== main paths ==============================

		t.Run("ReLU model with W=-2 B=-1 / Predict before Fit / returns [3, 1, 0, 0]", func(t *testing.T) {
			// ----- given -----
			wInitializer := initializers.NewFull(&initializers.FullConfig{Value: -2.})
			bInitializer := initializers.NewFull(&initializers.FullConfig{Value: -1.})

			input := stream.Input()
			hidden := stream.FC(&layers.FCConfig{
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Weight": wInitializer,
					"Bias":   bInitializer,
				},
				Device: dev,
			})(input)
			output := stream.Relu()(hidden)

			optimizer, err := optimizers.NewSGD(&optimizers.SGDConfig{LearningRate: 0.5})
			if err != nil {
				t.Fatal(err)
			}

			m, err := model.NewModel(input, output, &model.ModelConfig{
				Loss:      losses.NewMSE(),
				Optimizer: optimizer,
			})
			if err != nil {
				t.Fatal(err)
			}

			// ----- when -----
			x, err := tensor.Of([][]float64{{-2.}, {-1.}, {1.}, {2.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := m.Predict([]tensor.Tensor{x})
			if err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			exp, err := tensor.Of([][]float64{{3.}, {1.}, {0.}, {0.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("ReLU model with W=-2 B=-1 after Fit for 2 epochs / Predict / returns [0, 0, 0, 0]", func(t *testing.T) {
			// ----- given -----
			wInitializer := initializers.NewFull(&initializers.FullConfig{Value: -2.})
			bInitializer := initializers.NewFull(&initializers.FullConfig{Value: -1.})

			input := stream.Input()
			hidden := stream.FC(&layers.FCConfig{
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Weight": wInitializer,
					"Bias":   bInitializer,
				},
				Device: dev,
			})(input)
			output := stream.Relu()(hidden)

			optimizer, err := optimizers.NewSGD(&optimizers.SGDConfig{LearningRate: 0.5})
			if err != nil {
				t.Fatal(err)
			}

			m, err := model.NewModel(input, output, &model.ModelConfig{
				Loss:      losses.NewMSE(),
				Optimizer: optimizer,
			})
			if err != nil {
				t.Fatal(err)
			}

			// ----- when -----
			x := [][]float64{{-2.}, {-1.}, {1.}, {2.}}
			y := [][]float64{{0.}, {0.}, {1.}, {2.}}

			batchGen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
				BatchSize: 2,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			err = m.Fit(batchGen, batchGen, &model.FitConfig{Epochs: 2})
			if err != nil {
				t.Fatal(err)
			}

			xt, err := tensor.Of(x, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			act, err := m.Predict([]tensor.Tensor{xt})
			if err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			exp, err := tensor.Of([][]float64{{0.}, {0.}, {0.}, {0.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := act.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("ReLU model with W=-2 B=-1 / Eval / cumulative MSE goes 3.75", func(t *testing.T) {
			// ----- given -----
			wInitializer := initializers.NewFull(&initializers.FullConfig{Value: -2.})
			bInitializer := initializers.NewFull(&initializers.FullConfig{Value: -1.})

			input := stream.Input()
			hidden := stream.FC(&layers.FCConfig{
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Weight": wInitializer,
					"Bias":   bInitializer,
				},
				Device: dev,
			})(input)
			output := stream.Relu()(hidden)

			optimizer, err := optimizers.NewSGD(&optimizers.SGDConfig{LearningRate: 0.5})
			if err != nil {
				t.Fatal(err)
			}

			m, err := model.NewModel(input, output, &model.ModelConfig{
				Loss:      losses.NewMSE(),
				Optimizer: optimizer,
			})
			if err != nil {
				t.Fatal(err)
			}

			// ----- when -----
			x := [][]float64{{-2.}, {-1.}, {1.}, {2.}}
			y := [][]float64{{0.}, {0.}, {1.}, {2.}}
			evalMetrics := map[string]model.Metric{
				"MSE": metrics.NewMSE(),
			}

			batchGen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
				BatchSize: 2,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			result, err := m.Eval(batchGen, evalMetrics)
			if err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			if val := result["MSE"]; !(3.74-1e-10 < val && val < 3.76+1e-10) {
				t.Fatalf("expected metric value to be (3.75): got (%f)", val)
			}
		})

		t.Run("ReLU model with W=-2 B=-1 / Eval after Fit / cumulative MSE goes 1.25", func(t *testing.T) {
			// ----- given -----
			wInitializer := initializers.NewFull(&initializers.FullConfig{Value: -2.})
			bInitializer := initializers.NewFull(&initializers.FullConfig{Value: -1.})

			input := stream.Input()
			hidden := stream.FC(&layers.FCConfig{
				Outputs: 1,
				Initializers: map[string]layers.Initializer{
					"Weight": wInitializer,
					"Bias":   bInitializer,
				},
				Device: dev,
			})(input)
			output := stream.Relu()(hidden)

			optimizer, err := optimizers.NewSGD(&optimizers.SGDConfig{LearningRate: 0.5})
			if err != nil {
				t.Fatal(err)
			}

			m, err := model.NewModel(input, output, &model.ModelConfig{
				Loss:      losses.NewMSE(),
				Optimizer: optimizer,
			})
			if err != nil {
				t.Fatal(err)
			}

			// ----- when -----
			x := [][]float64{{-2.}, {-1.}, {1.}, {2.}}
			y := [][]float64{{0.}, {0.}, {1.}, {2.}}
			evalMetrics := map[string]model.Metric{
				"MSE": metrics.NewMSE(),
			}

			batchGen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
				BatchSize: 2,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			err = m.Fit(batchGen, batchGen, &model.FitConfig{Epochs: 2})
			if err != nil {
				t.Fatal(err)
			}

			result, err := m.Eval(batchGen, evalMetrics)
			if err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			if val := result["MSE"]; !(1.24-1e-10 < val && val < 1.26+1e-10) {
				t.Fatalf("expected metric value to be (1.25): got (%f)", val)
			}
		})

		// ============================== error handling ==============================

		t.Run("FC layer connected to 2 inputs / Fit / returns forward validation error", func(t *testing.T) {
			// ----- given -----
			input := stream.Input()
			hidden := stream.FC(&layers.FCConfig{
				Outputs: 1,
				Device:  dev,
			})(input, input)
			output := stream.Tanh()(hidden)

			optimizer, err := optimizers.NewSGD(nil)
			if err != nil {
				t.Fatal(err)
			}

			m, err := model.NewModel(input, output, &model.ModelConfig{
				Loss:      losses.NewMSE(),
				Optimizer: optimizer,
			})
			if err != nil {
				t.Fatal(err)
			}

			// ----- when -----
			x := [][]float64{{0.}}
			y := [][]float64{{0.}}

			batchGen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
				BatchSize: 1,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			err = m.Fit(batchGen, nil, &model.FitConfig{Epochs: 1})
			if err == nil {
				t.Fatal("expected error because of feed-forward validation")
			} else if err.Error() != "Fit operation failed: (Layer 1): forward operation on node: FC input data validation failed: expected exactly one input tensor: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("FC layer connected to 2 inputs / Eval / returns forward validation error", func(t *testing.T) {
			// ----- given -----
			input := stream.Input()
			hidden := stream.FC(&layers.FCConfig{
				Outputs: 1,
				Device:  dev,
			})(input, input)
			output := stream.Tanh()(hidden)

			optimizer, err := optimizers.NewSGD(nil)
			if err != nil {
				t.Fatal(err)
			}

			m, err := model.NewModel(input, output, &model.ModelConfig{
				Loss:      losses.NewMSE(),
				Optimizer: optimizer,
			})
			if err != nil {
				t.Fatal(err)
			}

			// ----- when -----
			x := [][]float64{{0.}}
			y := [][]float64{{0.}}

			batchGen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
				BatchSize: 1,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			_, err = m.Eval(batchGen, nil)
			if err == nil {
				t.Fatal("expected error because of feed-forward validation")
			} else if err.Error() != "Eval operation failed: Predict operation failed: (Layer 1): forward operation on node: FC input data validation failed: expected exactly one input tensor: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("2-output model with 1-output training labels / Fit / returns MSE loss dimension mismatch error", func(t *testing.T) {
			// ----- given -----
			input := stream.Input()
			hidden := stream.FC(&layers.FCConfig{
				Outputs: 2,
				Device:  dev,
			})(input)
			output := stream.Tanh()(hidden)

			optimizer, err := optimizers.NewSGD(nil)
			if err != nil {
				t.Fatal(err)
			}

			m, err := model.NewModel(input, output, &model.ModelConfig{
				Loss:      losses.NewMSE(),
				Optimizer: optimizer,
			})
			if err != nil {
				t.Fatal(err)
			}

			// ----- when -----
			x := [][]float64{{0.}}
			y := [][]float64{{0.}}

			batchgen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
				BatchSize: 1,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			err = m.Fit(batchgen, nil, &model.FitConfig{Epochs: 1})
			if err == nil {
				t.Fatal("expected error because of loss validation")
			} else if err.Error() != "Fit operation failed: MSE input data validation failed: expected input tensor sizes to match along data dimension: (2) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("2-output model with 1-output test labels / Eval with MSE metric / returns metric dimension mismatch error", func(t *testing.T) {
			// ----- given -----
			input := stream.Input()
			hidden := stream.FC(&layers.FCConfig{
				Outputs: 2,
				Device:  dev,
			})(input)
			output := stream.Tanh()(hidden)

			optimizer, err := optimizers.NewSGD(nil)
			if err != nil {
				t.Fatal(err)
			}

			m, err := model.NewModel(input, output, &model.ModelConfig{
				Loss:      losses.NewMSE(),
				Optimizer: optimizer,
			})
			if err != nil {
				t.Fatal(err)
			}

			// ----- when -----
			x := [][]float64{{0.}}
			y := [][]float64{{0.}}

			batchgen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
				BatchSize: 1,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			_, err = m.Eval(batchgen, map[string]model.Metric{"MSE": metrics.NewMSE()})
			if err == nil {
				t.Fatal("expected error because of metric validation")
			} else if err.Error() != "Eval operation failed: MSE input data validation failed: expected input tensor sizes to match along data dimension: (2) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("2-output model with 1-output validation labels and MSE metric / Fit / returns metric dimension mismatch error during validation", func(t *testing.T) {
			// ----- given -----
			input := stream.Input()
			hidden := stream.FC(&layers.FCConfig{
				Outputs: 2,
				Device:  dev,
			})(input)
			output := stream.Tanh()(hidden)

			optimizer, err := optimizers.NewSGD(nil)
			if err != nil {
				t.Fatal(err)
			}

			m, err := model.NewModel(input, output, &model.ModelConfig{
				Loss:      losses.NewMSE(),
				Optimizer: optimizer,
			})
			if err != nil {
				t.Fatal(err)
			}

			// ----- when -----
			x := [][]float64{{0.}}
			y2 := [][]float64{{0., 0.}}
			y1 := [][]float64{{0.}}

			trainBatchGen, err := batchgens.NewSimple(x, y2, &batchgens.SimpleConfig{
				BatchSize: 1,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			validBatchGen, err := batchgens.NewSimple(x, y1, &batchgens.SimpleConfig{
				BatchSize: 1,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			err = m.Fit(trainBatchGen, validBatchGen, &model.FitConfig{
				Epochs:  1,
				Metrics: map[string]model.Metric{"MSE": metrics.NewMSE()},
			})
			if err == nil {
				t.Fatal("expected error because of metric validation")
			} else if err.Error() != "Fit operation failed: Eval operation failed: MSE input data validation failed: expected input tensor sizes to match along data dimension: (2) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		// ============================== validations ==============================

		t.Run("NewMultiInputModel with nil config / returns error: config not to be nil", func(t *testing.T) {
			var (
				conf   *model.ModelConfig = nil
				input  *stream.Stream     = nil
				output *stream.Stream     = nil
				inputs                    = []*stream.Stream{input}
			)

			_, err := model.NewMultiInputModel(inputs, output, conf)
			if err == nil {
				t.Fatal("expected error because of nil input config")
			} else if err.Error() != "Model initialization failed: expected config not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewMultiInputModel with no input streams / returns error: at least one input stream", func(t *testing.T) {
			var (
				conf                    = new(model.ModelConfig)
				output *stream.Stream   = nil
				inputs []*stream.Stream = nil
			)

			_, err := model.NewMultiInputModel(inputs, output, conf)
			if err == nil {
				t.Fatal("expected error because of not having at least one input stream")
			} else if err.Error() != "Model initialization failed: expected to have at least one input stream" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewMultiInputModel with nil input stream at position 0 / returns error: input stream at 0 not to be nil", func(t *testing.T) {
			var (
				conf                  = new(model.ModelConfig)
				input  *stream.Stream = nil
				output *stream.Stream = nil
				inputs                = []*stream.Stream{input}
			)

			_, err := model.NewMultiInputModel(inputs, output, conf)
			if err == nil {
				t.Fatal("expected error because of receiving nil input stream")
			} else if err.Error() != "Model initialization failed: expected input stream at position (0) not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewMultiInputModel with uninitialized input stream at position 0 / returns error: input at 0 not properly initialized", func(t *testing.T) {
			var (
				conf                  = new(model.ModelConfig)
				input                 = new(stream.Stream)
				output *stream.Stream = nil
				inputs                = []*stream.Stream{input}
			)

			_, err := model.NewMultiInputModel(inputs, output, conf)
			if err == nil {
				t.Fatal("expected error because of receiving input stream which is not properly initialized")
			} else if err.Error() != "Model initialization failed: expected input stream at position (0) to be proparely initialized" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewMultiInputModel with non-Input type layer at position 0 / returns error: input at 0 must contain Input layer type", func(t *testing.T) {
			var (
				conf                  = new(model.ModelConfig)
				input                 = stream.Tanh()()
				output *stream.Stream = nil
				inputs                = []*stream.Stream{input}
			)

			_, err := model.NewMultiInputModel(inputs, output, conf)
			if err == nil {
				t.Fatal("expected error because of receiving input stream which does not contain layer of type 'Input'")
			} else if err.Error() != "Model initialization failed: expected input stream at position (0) to contain layer of type 'Input'" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewMultiInputModel with nil output stream / returns error: output stream not to be nil", func(t *testing.T) {
			var (
				conf                  = new(model.ModelConfig)
				input                 = stream.Input()
				output *stream.Stream = nil
				inputs                = []*stream.Stream{input}
			)

			_, err := model.NewMultiInputModel(inputs, output, conf)
			if err == nil {
				t.Fatal("expected error because of receiving nil output stream")
			} else if err.Error() != "Model initialization failed: expected output stream not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewMultiInputModel with uninitialized output stream / returns error: output stream not properly initialized", func(t *testing.T) {
			var (
				conf   = new(model.ModelConfig)
				input  = stream.Input()
				output = new(stream.Stream)
				inputs = []*stream.Stream{input}
			)

			_, err := model.NewMultiInputModel(inputs, output, conf)
			if err == nil {
				t.Fatal("expected error because of receiving output stream which is not properly initialized")
			} else if err.Error() != "Model initialization failed: expected output stream to be proparely initialized" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewMultiInputModel with Softmax Dim=-1 / returns error: Softmax Dim not to be negative", func(t *testing.T) {
			var (
				conf   = new(model.ModelConfig)
				input  = stream.Input()
				output = stream.Softmax(&activations.SoftmaxConfig{
					Dim: -1,
				})(input)
				inputs = []*stream.Stream{input}
			)

			_, err := model.NewMultiInputModel(inputs, output, conf)
			if err == nil {
				t.Fatal("expected error because of receiving output stream which is not properly initialized")
			} else if err.Error() != "Model initialization failed: \n(Layer 1): Softmax config data validation failed: expected 'Dim' not to be negative: got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Fit with nil config / returns error: Fit config not to be nil", func(t *testing.T) {
			input := stream.Input()
			output := stream.Softmax(&activations.SoftmaxConfig{Dim: 0})(input)
			m, err := model.NewMultiInputModel([]*stream.Stream{input}, output, new(model.ModelConfig))
			if err != nil {
				t.Fatal(err)
			}

			err = m.Fit(nil, nil, nil)
			if err == nil {
				t.Fatal("expected error because of nil Fit config")
			} else if err.Error() != "Fit config data validation failed: expected config not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Fit with 0 epochs / returns error: Epochs to be positive", func(t *testing.T) {
			input := stream.Input()
			output := stream.Softmax(&activations.SoftmaxConfig{Dim: 0})(input)
			m, err := model.NewMultiInputModel([]*stream.Stream{input}, output, new(model.ModelConfig))
			if err != nil {
				t.Fatal(err)
			}

			err = m.Fit(nil, nil, &model.FitConfig{Epochs: 0})
			if err == nil {
				t.Fatal("expected error because of 0 epochs")
			} else if err.Error() != "Fit config data validation failed: expected 'Epochs' to be positive: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("Predict with 2 inputs for 1-input model / returns error: expected exactly 1 input tensor", func(t *testing.T) {
			input := stream.Input()
			output := stream.Softmax(&activations.SoftmaxConfig{Dim: 0})(input)
			m, err := model.NewMultiInputModel([]*stream.Stream{input}, output, new(model.ModelConfig))
			if err != nil {
				t.Fatal(err)
			}

			_, err = m.Predict([]tensor.Tensor{nil, nil})
			if err == nil {
				t.Fatal("expected error because of wrong number of input tensors")
			} else if err.Error() != "Predict operation failed: expected exactly (1) input tensors: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
