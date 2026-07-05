package stream_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/layers/activations"
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
			x = stream.Softmax(&activations.SoftmaxConfig{Dim: 1})(x)

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

		t.Run("Add() / applying to two streams / no error", func(t *testing.T) {
			x := stream.Input()
			y := stream.Input()
			z := stream.Add()(x, y)

			if err := z.Error(); err != nil {
				t.Fatal(err)
			}
		})

		t.Run("Concat(valid config) / applying to two streams / no error", func(t *testing.T) {
			x := stream.Input()
			y := stream.Input()
			z := stream.Concat(&layers.ConcatConfig{Dim: 0})(x, y)

			if err := z.Error(); err != nil {
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
			x = stream.BatchNorm(nil)(x)

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
