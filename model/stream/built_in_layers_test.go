package stream_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/model/stream"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestBuiltInLayers(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Input() / creating input stream / no error", func(t *testing.T) {
			x := stream.Input()

			if err := x.Error(); err != nil {
				t.Fatal(err)
			}
		})

		t.Run("Tanh() / applying to stream / no error", func(t *testing.T) {
			x := stream.Input()
			x = stream.Tanh()(x)

			if err := x.Error(); err != nil {
				t.Fatal(err)
			}
		})

		t.Run("Sigmoid() / applying to stream / no error", func(t *testing.T) {
			x := stream.Input()
			x = stream.Sigmoid()(x)

			if err := x.Error(); err != nil {
				t.Fatal(err)
			}
		})

		t.Run("Softmax(nil) / applying to stream with nil config / no error", func(t *testing.T) {
			x := stream.Input()
			x = stream.Softmax(nil)(x)

			if err := x.Error(); err != nil {
				t.Fatal(err)
			}
		})

		t.Run("Relu() / applying to stream / no error", func(t *testing.T) {
			x := stream.Input()
			x = stream.Relu()(x)

			if err := x.Error(); err != nil {
				t.Fatal(err)
			}
		})

		t.Run("LeakyRelu(nil) / applying to stream with nil config / no error", func(t *testing.T) {
			x := stream.Input()
			x = stream.LeakyRelu(nil)(x)

			if err := x.Error(); err != nil {
				t.Fatal(err)
			}
		})

		t.Run("Dropout(nil) / applying to stream with nil config / no error", func(t *testing.T) {
			x := stream.Input()
			x = stream.Dropout(nil)(x)

			if err := x.Error(); err != nil {
				t.Fatal(err)
			}
		})

		t.Run("BatchNorm(valid config) / applying to stream / no error", func(t *testing.T) {
			x := stream.Input()
			x = stream.BatchNorm(&layers.BatchNormConfig{Momentum: 0.99, Eps: 1e-3})(x)

			if err := x.Error(); err != nil {
				t.Fatal(err)
			}
		})

		t.Run("FC(valid config) / applying to stream / no error", func(t *testing.T) {
			x := stream.Input()
			x = stream.FC(&layers.FCConfig{Outputs: 1})(x)

			if err := x.Error(); err != nil {
				t.Fatal(err)
			}
		})
	})
}
