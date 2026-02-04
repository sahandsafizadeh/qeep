package layers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestBatchNorm(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		confU := &tensor.Config{
			Device:    dev,
			GradTrack: false,
		}

		confT := &tensor.Config{
			Device:    dev,
			GradTrack: true,
		}

		/* ------------------------------ */

		layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
			Eps:    1e-10,
			Device: dev,
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err := tensor.Of([][]float64{{3.}}, confU)
		if err != nil {
			t.Fatal(err)
		}

		act, err := layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		expl, err := tensor.Of([][]float64{{2.9998}}, confU)
		if err != nil {
			t.Fatal(err)
		}

		expu, err := tensor.Of([][]float64{{3.}}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if p, err := act.Gt(expl); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		if p, err := act.Lt(expu); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		x, err = tensor.Of([][]float64{{3.}}, confT)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		expl, err = tensor.Of([][]float64{{-0.0001}}, confU)
		if err != nil {
			t.Fatal(err)
		}

		expu, err = tensor.Of([][]float64{{0.0001}}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if p, err := act.Gt(expl); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		if p, err := act.Lt(expu); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		layer, err = layers.NewBatchNorm(&layers.BatchNormConfig{
			Momentum: 0.5,
			Eps:      0.5,
			Device:   dev,
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err = tensor.Of([][]float64{{3., 1., 2.}}, confT)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		expl, err = tensor.Of([][]float64{{-0.0001, -0.0001, -0.0001}}, confU)
		if err != nil {
			t.Fatal(err)
		}

		expu, err = tensor.Of([][]float64{{0.0001, 0.0001, 0.0001}}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if p, err := act.Gt(expl); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		if p, err := act.Lt(expu); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		x, err = tensor.Of([][]float64{{3., 1., 2.}}, confU)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		expl, err = tensor.Of([][]float64{{1.4999, 0.4999, 0.9998}}, confU)
		if err != nil {
			t.Fatal(err)
		}

		expu, err = tensor.Of([][]float64{{1.5001, 0.5001, 1.0001}}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if p, err := act.Gt(expl); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		if p, err := act.Lt(expu); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		layer, err = layers.NewBatchNorm(&layers.BatchNormConfig{
			Momentum: 0.5,
			Eps:      1e-10,
			Device:   dev,
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err = tensor.Of([][]float64{
			{3., 0., 1., 2.},
			{1., 0., 2., 3.},
			{2., 0., 3., 1.},
		}, confT)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		expl, err = tensor.Of([][]float64{
			{0.9998, -0.0001, -1.0001, -0.0001},
			{-1.0001, -0.0001, -0.0001, 0.9998},
			{-0.0001, -0.0001, 0.9998, -1.0001},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		expu, err = tensor.Of([][]float64{
			{1.001, 0.0001, -0.9998, 0.0001},
			{-0.9998, 0.0001, 0.0001, 1.0001},
			{0.0001, 0.0001, 1.0001, -0.9998},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if p, err := act.Gt(expl); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		if p, err := act.Lt(expu); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		x, err = tensor.Of([][]float64{
			{6., 0., 2., 4.},
			{2., 0., 4., 6.},
		}, confT)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		expl, err = tensor.Of([][]float64{
			{0.7070, -0.0001, -0.7072, -0.7072},
			{-0.7072, -0.0001, 0.7070, 0.7070},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		expu, err = tensor.Of([][]float64{
			{0.7072, 0.0001, -0.7070, -0.7070},
			{-0.7070, 0.0001, 0.7072, 0.7072},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if p, err := act.Gt(expl); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		if p, err := act.Lt(expu); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		x, err = tensor.Of([][]float64{
			{0., 0., 0., 0.},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		expl, err = tensor.Of([][]float64{
			{-1.1786, -0.0001, -1.6330, -2.4495},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		expu, err = tensor.Of([][]float64{
			{-1.1785, 0.0001, -1.6329, -2.4494},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if p, err := act.Gt(expl); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		if p, err := act.Lt(expu); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		layer, err = layers.NewBatchNorm(&layers.BatchNormConfig{
			Momentum: 0.5,
			Eps:      1e-10,
			Device:   dev,
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err = tensor.Of([][][]float64{
			{
				{3., 0., 1., 2.},
				{1., 0., 2., 3.},
				{2., 0., 3., 1.},
			},
			{
				{3., 0., 1., 2.},
				{1., 0., 2., 3.},
				{2., 0., 3., 1.},
			},
		}, confT)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		expl, err = tensor.Of([][][]float64{
			{
				{1.1179, -0.0001, -1.1181, -0.0001},
				{-1.1181, -0.0001, -0.0001, 1.1179},
				{-0.0001, -0.0001, 1.1179, -1.1181},
			},
			{
				{1.1179, -0.0001, -1.1181, -0.0001},
				{-1.1181, -0.0001, -0.0001, 1.1179},
				{-0.0001, -0.0001, 1.1179, -1.1181},
			},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		expu, err = tensor.Of([][][]float64{
			{
				{1.1181, 0.0001, -1.1179, 0.0001},
				{-1.1179, 0.0001, 0.0001, 1.1181},
				{0.0001, 0.0001, 1.1181, -1.1179},
			},
			{
				{1.1181, 0.0001, -1.1179, 0.0001},
				{-1.1179, 0.0001, 0.0001, 1.1181},
				{0.0001, 0.0001, 1.1181, -1.1179},
			},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if p, err := act.Gt(expl); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		if p, err := act.Lt(expu); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		x, err = tensor.Of([][][]float64{
			{{0., 0., 0., 0.}},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		expl, err = tensor.Of([][][]float64{
			{{-1.0541, -0.0001, -1.0541, -1.0541}},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		expu, err = tensor.Of([][][]float64{
			{{-1.0539, 0.0001, -1.0539, -1.0539}},
		}, confU)
		if err != nil {
			t.Fatal(err)
		}

		if p, err := act.Gt(expl); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		if p, err := act.Lt(expu); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		layer, err = layers.NewBatchNorm(&layers.BatchNormConfig{
			Momentum: 0.5,
			Eps:      1e-10,
			Device:   dev,
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err = tensor.Ones([]int{8, 32, 32, 3}, confT)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		expl, err = tensor.Full([]int{8, 32, 32, 3}, -0.0001, confU)
		if err != nil {
			t.Fatal(err)
		}

		expu, err = tensor.Full([]int{8, 32, 32, 3}, 0.0001, confU)
		if err != nil {
			t.Fatal(err)
		}

		if p, err := act.Gt(expl); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		if p, err := act.Lt(expu); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		/* --------------- */

		x, err = tensor.Zeros([]int{1, 32, 32, 3}, confU)
		if err != nil {
			t.Fatal(err)
		}

		act, err = layer.Forward(x)
		if err != nil {
			t.Fatal(err)
		}

		expl, err = tensor.Full([]int{1, 32, 32, 3}, -0.7072, confU)
		if err != nil {
			t.Fatal(err)
		}

		expu, err = tensor.Full([]int{1, 32, 32, 3}, -0.7070, confU)
		if err != nil {
			t.Fatal(err)
		}

		if p, err := act.Gt(expl); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		if p, err := act.Lt(expu); err != nil {
			t.Fatal(err)
		} else if p.Sum() < float64(p.NElems()) {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		weights := layer.Weights()
		if len(weights) != 4 {
			t.Fatalf("expected BatchNorm to have (4) weights: got (%d)", len(weights))
		}

		expb := layers.Weight{
			Value:     &layer.Beta,
			Trainable: true,
		}
		if weights[0] != expb {
			t.Fatal("expected BatchNorm weight (0) to be trainable and point to 'Beta'")
		}

		expg := layers.Weight{
			Value:     &layer.Gamma,
			Trainable: true,
		}
		if weights[1] != expg {
			t.Fatal("expected BatchNorm weight (1) to be trainable and point to 'Gamma'")
		}

		expmm := layers.Weight{
			Value:     &layer.MovingMean,
			Trainable: false,
		}
		if weights[2] != expmm {
			t.Fatal("expected BatchNorm weight (2) to be trainable and point to 'MovingMean'")
		}

		expmv := layers.Weight{
			Value:     &layer.MovingVar,
			Trainable: false,
		}
		if weights[3] != expmv {
			t.Fatal("expected BatchNorm weight (3) to be trainable and point to 'MovingVar'")
		}

		/* ------------------------------ */

	})
}

func TestValidationBatchNorm(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		_, err := layers.NewBatchNorm(nil)
		if err == nil {
			t.Fatalf("expected error because of nil input config")
		} else if err.Error() != "BatchNorm config data validation failed: expected config not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layers.NewBatchNorm(&layers.BatchNormConfig{
			Momentum: -0.01,
		})
		if err == nil {
			t.Fatalf("expected error because of negative 'Momentum'")
		} else if err.Error() != "BatchNorm config data validation failed: expected 'Momentum' not to be negative: got (-0.010000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layers.NewBatchNorm(&layers.BatchNormConfig{
			Momentum: 0,
			Eps:      -0.001,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'Eps'")
		} else if err.Error() != "BatchNorm config data validation failed: expected 'Eps' to be positive: got (-0.001000)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layers.NewBatchNorm(&layers.BatchNormConfig{
			Momentum: 0,
			Eps:      0,
		})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'Eps'")
		} else if err.Error() != "BatchNorm config data validation failed: expected 'Eps' to be positive: got (0.000000)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
			Momentum: layers.BatchNormDefaultMomentum,
			Eps:      layers.BatchNormDefaultEps,
		})
		if err != nil {
			t.Fatal(err)
		}

		x, err := tensor.Zeros(nil, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = layer.Forward()
		if err == nil {
			t.Fatalf("expected error because of not receiving one input tensor")
		} else if err.Error() != "BatchNorm input data validation failed: expected exactly one input tensor: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layer.Forward(x, x)
		if err == nil {
			t.Fatalf("expected error because of not receiving one input tensor")
		} else if err.Error() != "BatchNorm input data validation failed: expected exactly one input tensor: got (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = layer.Forward(x)
		if err == nil {
			t.Fatalf("expected error because of not receiving one input tensor")
		} else if err.Error() != "BatchNorm input data validation failed: expected input tensor to have at least two dimensions (batch, ..., feature): got (0)" {
			t.Fatal("unexpected error message returned")
		}

		x, err = tensor.Zeros([]int{4}, conf)
		if err != nil {
			t.Fatal(err)
		}

		_, err = layer.Forward(x)
		if err == nil {
			t.Fatalf("expected error because of not receiving one input tensor")
		} else if err.Error() != "BatchNorm input data validation failed: expected input tensor to have at least two dimensions (batch, ..., feature): got (1)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

	})
}
