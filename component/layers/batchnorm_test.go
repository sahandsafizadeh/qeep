package layers_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestBatchNorm(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("BatchNorm(eps=1e-10) / Forward([[3]]) without grad track / output near 3", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Eps:    1e-10,
				Device: dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{{3.}}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of([][]float64{{2.9998}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of([][]float64{{3.}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("BatchNorm(eps=1e-10) / Forward([[3]]) with grad track / output near 0", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Eps:    1e-10,
				Device: dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{{3.}}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of([][]float64{{-0.0001}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of([][]float64{{0.0001}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("BatchNorm(momentum=0.5, eps=0.5) / Forward([[3,1,2]]) with grad track / output near 0", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: 0.5,
				Eps:      0.5,
				Device:   dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{{3., 1., 2.}}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of([][]float64{{-0.0001, -0.0001, -0.0001}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of([][]float64{{0.0001, 0.0001, 0.0001}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("BatchNorm(momentum=0.5, eps=0.5) / Forward([[3,1,2]]) without grad track / output uses running stats", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: 0.5,
				Eps:      0.5,
				Device:   dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			// tracked pass to update running stat
			xinit, err := tensor.Of([][]float64{{3., 1., 2.}}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = layer.Forward(xinit)
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{{3., 1., 2.}}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of([][]float64{{1.4999, 0.4999, 0.9998}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of([][]float64{{1.5001, 0.5001, 1.0001}}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("BatchNorm(momentum=0.5, eps=1e-10) / Forward([3x4]) with grad track / normalizes across batch", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: 0.5,
				Eps:      1e-10,
				Device:   dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{
				{3., 0., 1., 2.},
				{1., 0., 2., 3.},
				{2., 0., 3., 1.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of([][]float64{
				{0.9998, -0.0001, -1.0001, -0.0001},
				{-1.0001, -0.0001, -0.0001, 0.9998},
				{-0.0001, -0.0001, 0.9998, -1.0001},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of([][]float64{
				{1.001, 0.0001, -0.9998, 0.0001},
				{-0.9998, 0.0001, 0.0001, 1.0001},
				{0.0001, 0.0001, 1.0001, -0.9998},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("BatchNorm(momentum=0.5, eps=1e-10) / Forward([2x4]) with grad track / output near ±0.707", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: 0.5,
				Eps:      1e-10,
				Device:   dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{
				{6., 0., 2., 4.},
				{2., 0., 4., 6.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of([][]float64{
				{0.7070, -0.0001, -0.7072, -0.7072},
				{-0.7072, -0.0001, 0.7070, 0.7070},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of([][]float64{
				{0.7072, 0.0001, -0.7070, -0.7070},
				{-0.7070, 0.0001, 0.7072, 0.7072},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("BatchNorm(momentum=0.5, eps=1e-10) / Forward([[0,0,0,0]]) without grad track after two grad passes / output uses running stats", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: 0.5,
				Eps:      1e-10,
				Device:   dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			// tracked pass to update running stat
			xinit1, err := tensor.Of([][]float64{
				{3., 0., 1., 2.},
				{1., 0., 2., 3.},
				{2., 0., 3., 1.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = layer.Forward(xinit1)
			if err != nil {
				t.Fatal(err)
			}

			// tracked pass to update running stats
			xinit2, err := tensor.Of([][]float64{
				{6., 0., 2., 4.},
				{2., 0., 4., 6.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = layer.Forward(xinit2)
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{
				{0., 0., 0., 0.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of([][]float64{
				{-1.1786, -0.0001, -1.6330, -2.4495},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of([][]float64{
				{-1.1785, 0.0001, -1.6329, -2.4494},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("BatchNorm(momentum=0.5, eps=1e-10) / Forward([2x3x4]) 3D input with grad track / normalizes across batch and spatial dims", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: 0.5,
				Eps:      1e-10,
				Device:   dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("BatchNorm(momentum=0.5, eps=1e-10) / Forward([[[0,0,0,0]]]) without grad track after 3D grad pass / output uses running stats", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: 0.5,
				Eps:      1e-10,
				Device:   dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			// tracked pass to update running stats
			xinit, err := tensor.Of([][][]float64{
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
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = layer.Forward(xinit)
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][][]float64{
				{{0., 0., 0., 0.}},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of([][][]float64{
				{{-1.0541, -0.0001, -1.0541, -1.0541}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of([][][]float64{
				{{-1.0539, 0.0001, -1.0539, -1.0539}},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("BatchNorm(momentum=0.5, eps=1e-10) / Forward([8x32x32x3] ones) with grad track / output near 0", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: 0.5,
				Eps:      1e-10,
				Device:   dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Ones([]int{8, 32, 32, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Full([]int{8, 32, 32, 3}, -0.0001, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Full([]int{8, 32, 32, 3}, 0.0001, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("BatchNorm(momentum=0.5, eps=1e-10) / Forward([1x32x32x3] zeros) without grad track after ones pass / output near -0.707", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: 0.5,
				Eps:      1e-10,
				Device:   dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			// tracked pass to update running stats
			xinit, err := tensor.Ones([]int{8, 32, 32, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = layer.Forward(xinit)
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros([]int{1, 32, 32, 3}, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Full([]int{1, 32, 32, 3}, -0.7072, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Full([]int{1, 32, 32, 3}, -0.7070, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected output to be in range")
			}
		})

		t.Run("BatchNorm(default config nil) / train [[1,0],[-1,0]] then test [[0,0]] / test near 0", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(nil)
			if err != nil {
				t.Fatal(err)
			}

			// tracked pass to update running stat
			xinit, err := tensor.Of([][]float64{
				{1., 0.},
				{-1., 0.},
			}, &tensor.Config{
				Device:    tensor.CPU,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = layer.Forward(xinit)
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{{0., 0.}}, &tensor.Config{
				Device:    tensor.CPU,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of([][]float64{{-0.0001, -0.0001}}, &tensor.Config{Device: tensor.CPU})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of([][]float64{{0.0001, 0.0001}}, &tensor.Config{Device: tensor.CPU})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected test output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected test output to be in range")
			}
		})

		t.Run("BatchNorm(default config empty) / train [[1,0],[-1,0]] then test [[0,0]] / test near 0", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{})
			if err != nil {
				t.Fatal(err)
			}

			// tracked pass to update running stat
			xinit, err := tensor.Of([][]float64{
				{1., 0.},
				{-1., 0.},
			}, &tensor.Config{
				Device:    tensor.CPU,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = layer.Forward(xinit)
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Of([][]float64{{0., 0.}}, &tensor.Config{
				Device:    tensor.CPU,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			act, err := layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			expl, err := tensor.Of([][]float64{{-0.0001, -0.0001}}, &tensor.Config{Device: tensor.CPU})
			if err != nil {
				t.Fatal(err)
			}
			expu, err := tensor.Of([][]float64{{0.0001, 0.0001}}, &tensor.Config{Device: tensor.CPU})
			if err != nil {
				t.Fatal(err)
			}

			if p, err := act.Gt(expl); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected test output to be in range")
			}
			if p, err := act.Lt(expu); err != nil {
				t.Fatal(err)
			} else if p.Sum() < float64(p.NElems()) {
				t.Fatal("expected test output to be in range")
			}
		})

		t.Run("BatchNorm layer / Weights() / before first forward, all 4 weights are uninitialized", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: 0.5,
				Eps:      1e-10,
				Device:   dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			weights := layer.Weights()
			if len(weights) != 4 {
				t.Fatalf("expected BatchNorm to have (4) weights: got (%d)", len(weights))
			}

			if !weights[0].Trainable || *weights[0].Value != nil {
				t.Fatal("expected BatchNorm weight (0) to be trainable with nil value")
			}
			if !weights[1].Trainable || *weights[1].Value != nil {
				t.Fatal("expected BatchNorm weight (1) to be trainable with nil value")
			}
			if weights[2].Trainable || *weights[2].Value != nil {
				t.Fatal("expected BatchNorm weight (2) to be non-trainable with nil value")
			}
			if weights[3].Trainable || *weights[3].Value != nil {
				t.Fatal("expected BatchNorm weight (3) to be non-trainable with nil value")
			}
		})

		t.Run("BatchNorm layer / Weights() / after Forward(), returns 4 initialized weights pointing to Beta, Gamma, MovingMean, MovingVar", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: 0.5,
				Eps:      1e-10,
				Device:   dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Ones([]int{4, 8, 8}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			_, err = layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			weights := layer.Weights()
			if len(weights) != 4 {
				t.Fatalf("expected BatchNorm to have (4) weights: got (%d)", len(weights))
			}

			if !weights[0].Trainable || *weights[0].Value == nil || weights[0].Value != &layer.Beta {
				t.Fatal("expected BatchNorm weight (0) to be trainable, non-nil and point to 'Beta'")
			}
			if !weights[1].Trainable || *weights[1].Value == nil || weights[1].Value != &layer.Gamma {
				t.Fatal("expected BatchNorm weight (1) to be trainable, non-nil and point to 'Gamma'")
			}
			if weights[2].Trainable || *weights[2].Value == nil || weights[2].Value != &layer.MovingMean {
				t.Fatal("expected BatchNorm weight (2) to be non-trainable, non-nil and point to 'MovingMean'")
			}
			if weights[3].Trainable || *weights[3].Value == nil || weights[3].Value != &layer.MovingVar {
				t.Fatal("expected BatchNorm weight (3) to be non-trainable, non-nil and point to 'MovingVar'")
			}
		})

		t.Run("BatchNorm layer / Weights() / inference, pre-initialized weights stay the same after Forward()", func(t *testing.T) {
			// ----- given -----
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: 0.5,
				Eps:      1e-10,
				Device:   dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			b, err := tensor.Ones(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			g, err := tensor.Ones(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			mm, err := tensor.Ones(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			mv, err := tensor.Ones(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			layer.Beta = b
			layer.Gamma = g
			layer.MovingMean = mm
			layer.MovingVar = mv

			// ----- when -----
			x, err := tensor.Ones([]int{4, 8, 8}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			_, err = layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			weights := layer.Weights()
			if len(weights) != 4 {
				t.Fatalf("expected BatchNorm to have (4) weights: got (%d)", len(weights))
			}

			if !weights[0].Trainable || *weights[0].Value != b || weights[0].Value != &layer.Beta {
				t.Fatal("expected BatchNorm weight (0) to be trainable, stay the same and point to 'Beta'")
			}
			if !weights[1].Trainable || *weights[1].Value != g || weights[1].Value != &layer.Gamma {
				t.Fatal("expected BatchNorm weight (1) to be trainable, stay the same and point to 'Gamma'")
			}
			if weights[2].Trainable || *weights[2].Value != mm || weights[2].Value != &layer.MovingMean {
				t.Fatal("expected BatchNorm weight (2) to be non-trainable, stay the same and point to 'MovingMean'")
			}
			if weights[3].Trainable || *weights[3].Value != mv || weights[3].Value != &layer.MovingVar {
				t.Fatal("expected BatchNorm weight (3) to be non-trainable, stay the same and point to 'MovingVar'")
			}
		})

		t.Run("BatchNorm layer / Weights() / train, only trainable pre-initialized weights stay the same after Forward()", func(t *testing.T) {
			// ----- given -----
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: 0.5,
				Eps:      1e-10,
				Device:   dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			b, err := tensor.Ones(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			g, err := tensor.Ones(nil, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			mm, err := tensor.Ones(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			mv, err := tensor.Ones(nil, &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			layer.Beta = b
			layer.Gamma = g
			layer.MovingMean = mm
			layer.MovingVar = mv

			// ----- when -----
			x, err := tensor.Ones([]int{4, 8, 8}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = layer.Forward(x)
			if err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			weights := layer.Weights()
			if len(weights) != 4 {
				t.Fatalf("expected BatchNorm to have (4) weights: got (%d)", len(weights))
			}

			if !weights[0].Trainable || *weights[0].Value != b || weights[0].Value != &layer.Beta {
				t.Fatal("expected BatchNorm weight (0) to be trainable, stay the same and point to 'Beta'")
			}
			if !weights[1].Trainable || *weights[1].Value != g || weights[1].Value != &layer.Gamma {
				t.Fatal("expected BatchNorm weight (1) to be trainable, stay the same and point to 'Gamma'")
			}
			if weights[2].Trainable || *weights[2].Value == mm || weights[2].Value != &layer.MovingMean {
				t.Fatal("expected BatchNorm weight (2) to be non-trainable, change and point to 'MovingMean'")
			}
			if weights[3].Trainable || *weights[3].Value == mv || weights[3].Value != &layer.MovingVar {
				t.Fatal("expected BatchNorm weight (3) to be non-trainable, change and point to 'MovingVar'")
			}
		})

		// ============================== validations ==============================

		t.Run("NewBatchNorm(Momentum=-0.01) / returns error: Momentum must not be negative", func(t *testing.T) {
			_, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: -0.01,
			})
			if err == nil {
				t.Fatal("expected error because of negative 'Momentum'")
			} else if err.Error() != "BatchNorm config data validation failed: expected 'Momentum' to be positive: got (-0.010000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewBatchNorm(Eps=-0.001) / returns error: Eps must be positive", func(t *testing.T) {
			_, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: 0,
				Eps:      -0.001,
			})
			if err == nil {
				t.Fatal("expected error because of non-positive 'Eps'")
			} else if err.Error() != "BatchNorm config data validation failed: expected 'Eps' to be positive: got (-0.001000)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("BatchNorm.Forward() with no inputs / returns error: expected exactly one input tensor", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: layers.BatchNormDefaultMomentum,
				Eps:      layers.BatchNormDefaultEps,
			})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward()
			if err == nil {
				t.Fatal("expected error because of not receiving one input tensor")
			} else if err.Error() != "BatchNorm input data validation failed: expected exactly one input tensor: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("BatchNorm.Forward() with two inputs / returns error: expected exactly one input tensor", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: layers.BatchNormDefaultMomentum,
				Eps:      layers.BatchNormDefaultEps,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x, x)
			if err == nil {
				t.Fatal("expected error because of not receiving one input tensor")
			} else if err.Error() != "BatchNorm input data validation failed: expected exactly one input tensor: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("BatchNorm.Forward() with scalar input / returns error: expected at least two dimensions", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: layers.BatchNormDefaultMomentum,
				Eps:      layers.BatchNormDefaultEps,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros(nil, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x)
			if err == nil {
				t.Fatal("expected error because of scalar input")
			} else if err.Error() != "BatchNorm input data validation failed: expected input tensor to have at least two dimensions (batch, ..., feature): got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("BatchNorm.Forward() with 1D input / returns error: expected at least two dimensions", func(t *testing.T) {
			layer, err := layers.NewBatchNorm(&layers.BatchNormConfig{
				Momentum: layers.BatchNormDefaultMomentum,
				Eps:      layers.BatchNormDefaultEps,
			})
			if err != nil {
				t.Fatal(err)
			}

			x, err := tensor.Zeros([]int{4}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			_, err = layer.Forward(x)
			if err == nil {
				t.Fatal("expected error because of 1D input")
			} else if err.Error() != "BatchNorm input data validation failed: expected input tensor to have at least two dimensions (batch, ..., feature): got (1)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
