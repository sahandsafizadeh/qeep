package node_test

import (
	"slices"
	"testing"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/component/optimizers"
	"github.com/sahandsafizadeh/qeep/model/internal/node"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestNode(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("6-node graph (3 parents, 1 middle, 2 children) / Layer() / returns the layer passed to NewNode", func(t *testing.T) {
			lp1 := activations.NewTanh()
			lp2 := activations.NewTanh()
			lp3 := activations.NewTanh()
			lm1 := activations.NewRelu()
			lc1 := activations.NewSigmoid()
			lc2 := activations.NewSigmoid()

			np1 := node.NewNode(lp1)
			np2 := node.NewNode(lp2)
			np3 := node.NewNode(lp3)
			nm1 := node.NewNode(lm1)
			nc1 := node.NewNode(lc1)
			nc2 := node.NewNode(lc2)

			if np1.Layer() != lp1 {
				t.Fatal("unexpected node Layer")
			}
			if np2.Layer() != lp2 {
				t.Fatal("unexpected node Layer")
			}
			if np3.Layer() != lp3 {
				t.Fatal("unexpected node Layer")
			}
			if nm1.Layer() != lm1 {
				t.Fatal("unexpected node Layer")
			}
			if nc1.Layer() != lc1 {
				t.Fatal("unexpected node Layer")
			}
			if nc2.Layer() != lc2 {
				t.Fatal("unexpected node Layer")
			}
		})

		t.Run("6-node graph / Result() before any Forward / returns nil for all nodes", func(t *testing.T) {
			np1 := node.NewNode(activations.NewTanh())
			np2 := node.NewNode(activations.NewTanh())
			np3 := node.NewNode(activations.NewTanh())
			nm1 := node.NewNode(activations.NewRelu())
			nc1 := node.NewNode(activations.NewSigmoid())
			nc2 := node.NewNode(activations.NewSigmoid())

			if np1.Result() != nil {
				t.Fatal("unexpected node Result")
			}
			if np2.Result() != nil {
				t.Fatal("unexpected node Result")
			}
			if np3.Result() != nil {
				t.Fatal("unexpected node Result")
			}
			if nm1.Result() != nil {
				t.Fatal("unexpected node Result")
			}
			if nc1.Result() != nil {
				t.Fatal("unexpected node Result")
			}
			if nc2.Result() != nil {
				t.Fatal("unexpected node Result")
			}
		})

		t.Run("6-node graph with parents and children wired / Parents() / returns correct parent list for each node", func(t *testing.T) {
			np1 := node.NewNode(activations.NewTanh())
			np2 := node.NewNode(activations.NewTanh())
			np3 := node.NewNode(activations.NewTanh())
			nm1 := node.NewNode(activations.NewRelu())
			nc1 := node.NewNode(activations.NewSigmoid())
			nc2 := node.NewNode(activations.NewSigmoid())

			np1.AddChild(nm1)
			np2.AddChild(nm1)
			np3.AddChild(nm1)
			nm1.AddParent(np1)
			nm1.AddParent(np2)
			nm1.AddParent(np3)
			nm1.AddChild(nc1)
			nm1.AddChild(nc2)
			nc1.AddParent(nm1)
			nc2.AddParent(nm1)

			if !slices.Equal(np1.Parents(), nil) {
				t.Fatal("unexpected node Parents")
			}
			if !slices.Equal(np2.Parents(), nil) {
				t.Fatal("unexpected node Parents")
			}
			if !slices.Equal(np3.Parents(), nil) {
				t.Fatal("unexpected node Parents")
			}
			if !slices.Equal(nm1.Parents(), []*node.Node{np1, np2, np3}) {
				t.Fatal("unexpected node Parents")
			}
			if !slices.Equal(nc1.Parents(), []*node.Node{nm1}) {
				t.Fatal("unexpected node Parents")
			}
			if !slices.Equal(nc2.Parents(), []*node.Node{nm1}) {
				t.Fatal("unexpected node Parents")
			}
		})

		t.Run("6-node graph with parents and children wired / Children() / returns correct children list for each node", func(t *testing.T) {
			np1 := node.NewNode(activations.NewTanh())
			np2 := node.NewNode(activations.NewTanh())
			np3 := node.NewNode(activations.NewTanh())
			nm1 := node.NewNode(activations.NewRelu())
			nc1 := node.NewNode(activations.NewSigmoid())
			nc2 := node.NewNode(activations.NewSigmoid())

			np1.AddChild(nm1)
			np2.AddChild(nm1)
			np3.AddChild(nm1)
			nm1.AddParent(np1)
			nm1.AddParent(np2)
			nm1.AddParent(np3)
			nm1.AddChild(nc1)
			nm1.AddChild(nc2)
			nc1.AddParent(nm1)
			nc2.AddParent(nm1)

			if !slices.Equal(np1.Children(), []*node.Node{nm1}) {
				t.Fatal("unexpected node Children")
			}
			if !slices.Equal(np2.Children(), []*node.Node{nm1}) {
				t.Fatal("unexpected node Children")
			}
			if !slices.Equal(np3.Children(), []*node.Node{nm1}) {
				t.Fatal("unexpected node Children")
			}
			if !slices.Equal(nm1.Children(), []*node.Node{nc1, nc2}) {
				t.Fatal("unexpected node Children")
			}
			if !slices.Equal(nc1.Children(), nil) {
				t.Fatal("unexpected node Children")
			}
			if !slices.Equal(nc2.Children(), nil) {
				t.Fatal("unexpected node Children")
			}
		})

		t.Run("6-node graph with NLayer set / NLayer() / returns the value set by SetNLayer", func(t *testing.T) {
			np1 := node.NewNode(activations.NewTanh())
			np2 := node.NewNode(activations.NewTanh())
			np3 := node.NewNode(activations.NewTanh())
			nm1 := node.NewNode(activations.NewRelu())
			nc1 := node.NewNode(activations.NewSigmoid())
			nc2 := node.NewNode(activations.NewSigmoid())

			np1.SetNLayer(0)
			np2.SetNLayer(0)
			np3.SetNLayer(0)
			nm1.SetNLayer(1)
			nc1.SetNLayer(2)
			nc2.SetNLayer(2)

			if np1.NLayer() != 0 {
				t.Fatal("unexpected node NLayer")
			}
			if np2.NLayer() != 0 {
				t.Fatal("unexpected node NLayer")
			}
			if np3.NLayer() != 0 {
				t.Fatal("unexpected node NLayer")
			}
			if nm1.NLayer() != 1 {
				t.Fatal("unexpected node NLayer")
			}
			if nc1.NLayer() != 2 {
				t.Fatal("unexpected node NLayer")
			}
			if nc2.NLayer() != 2 {
				t.Fatal("unexpected node NLayer")
			}
		})

		t.Run("Input seeding [1,1], WeightedLayer(a=3, b=2), Relu, grad enabled / Forward on all nodes / input=[1,1], weighted=[6,6], activation=[6,6]", func(t *testing.T) {
			// ----- given -----
			input := layers.NewInput()
			input.SeedFunc = func() tensor.Tensor {
				x, err := tensor.Full([]int{2}, 1., &tensor.Config{Device: dev})
				if err != nil {
					t.Fatal(err)
				}
				return x
			}

			weighted, err := newSimpleWeightedLayer(&tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			activation := activations.NewRelu()

			ni := node.NewNode(input)
			nw := node.NewNode(weighted)
			na := node.NewNode(activation)

			ni.AddChild(nw)
			nw.AddParent(ni)
			nw.AddChild(na)
			na.AddParent(nw)

			ni.SetNLayer(0)
			nw.SetNLayer(1)
			na.SetNLayer(2)

			// ----- when -----
			ni.EnableGrad()
			nw.EnableGrad()
			na.EnableGrad()

			if err := ni.Forward(); err != nil {
				t.Fatal(err)
			}
			if err := nw.Forward(); err != nil {
				t.Fatal(err)
			}
			if err := na.Forward(); err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			exp, err := tensor.Full([]int{2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			if eq, err := ni.Result().Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			exp, err = tensor.Full([]int{2}, 6., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			if eq, err := nw.Result().Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			exp, err = tensor.Full([]int{2}, 6., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			if eq, err := na.Result().Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Input seeding [1,1], WeightedLayer(a=3, b=2), Relu / Forward + BackPropagate + Optimize(SGD lr=1) / trainable weight a updated from 3 to 2, non-trainable b stays at 2", func(t *testing.T) {
			// ----- given -----
			input := layers.NewInput()
			input.SeedFunc = func() tensor.Tensor {
				x, err := tensor.Full([]int{2}, 1., &tensor.Config{Device: dev})
				if err != nil {
					t.Fatal(err)
				}
				return x
			}

			weighted, err := newSimpleWeightedLayer(&tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			activation := activations.NewRelu()

			optimizer, err := optimizers.NewSGD(&optimizers.SGDConfig{LearningRate: 1.})
			if err != nil {
				t.Fatal(err)
			}

			ni := node.NewNode(input)
			nw := node.NewNode(weighted)
			na := node.NewNode(activation)

			ni.AddChild(nw)
			nw.AddParent(ni)
			nw.AddChild(na)
			na.AddParent(nw)

			ni.SetNLayer(0)
			nw.SetNLayer(1)
			na.SetNLayer(2)

			// ----- when -----
			ni.EnableGrad()
			nw.EnableGrad()
			na.EnableGrad()

			if err := ni.Forward(); err != nil {
				t.Fatal(err)
			}
			if err := nw.Forward(); err != nil {
				t.Fatal(err)
			}
			if err := na.Forward(); err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(na.Result())
			if err != nil {
				t.Fatal(err)
			}

			if err := ni.Optimize(optimizer); err != nil {
				t.Fatal(err)
			}
			if err := nw.Optimize(optimizer); err != nil {
				t.Fatal(err)
			}
			if err := na.Optimize(optimizer); err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			exp, err := tensor.Full([]int{2}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			if eq, err := weighted.a.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			exp, err = tensor.Full([]int{2}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			if eq, err := weighted.b.Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("Input seeding [1,1], WeightedLayer after optimization (a=2, b=2), Relu, grad disabled / Forward on all nodes / input=[1,1], weighted=[5,5], activation=[5,5]", func(t *testing.T) {
			// ----- given -----
			input := layers.NewInput()
			input.SeedFunc = func() tensor.Tensor {
				x, err := tensor.Full([]int{2}, 1., &tensor.Config{Device: dev})
				if err != nil {
					t.Fatal(err)
				}
				return x
			}

			weighted, err := newSimpleWeightedLayer(&tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			activation := activations.NewRelu()

			optimizer, err := optimizers.NewSGD(&optimizers.SGDConfig{LearningRate: 1.})
			if err != nil {
				t.Fatal(err)
			}

			ni := node.NewNode(input)
			nw := node.NewNode(weighted)
			na := node.NewNode(activation)

			ni.AddChild(nw)
			nw.AddParent(ni)
			nw.AddChild(na)
			na.AddParent(nw)

			ni.SetNLayer(0)
			nw.SetNLayer(1)
			na.SetNLayer(2)

			// ----- when -----
			ni.EnableGrad()
			nw.EnableGrad()
			na.EnableGrad()

			if err := ni.Forward(); err != nil {
				t.Fatal(err)
			}
			if err := nw.Forward(); err != nil {
				t.Fatal(err)
			}
			if err := na.Forward(); err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(na.Result())
			if err != nil {
				t.Fatal(err)
			}

			if err := ni.Optimize(optimizer); err != nil {
				t.Fatal(err)
			}
			if err := nw.Optimize(optimizer); err != nil {
				t.Fatal(err)
			}
			if err := na.Optimize(optimizer); err != nil {
				t.Fatal(err)
			}

			ni.DisableGrad()
			nw.DisableGrad()
			na.DisableGrad()

			if err := ni.Forward(); err != nil {
				t.Fatal(err)
			}
			if err := nw.Forward(); err != nil {
				t.Fatal(err)
			}
			if err := na.Forward(); err != nil {
				t.Fatal(err)
			}

			// ----- then -----
			exp, err := tensor.Full([]int{2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			if eq, err := ni.Result().Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			exp, err = tensor.Full([]int{2}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			if eq, err := nw.Result().Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			exp, err = tensor.Full([]int{2}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			if eq, err := na.Result().Equals(exp); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== error handling ==============================

		t.Run("FC node with 2 parent inputs / Forward / returns error: expected exactly one input tensor", func(t *testing.T) {
			input1 := layers.NewInput()
			input2 := layers.NewInput()

			input1.SeedFunc = func() tensor.Tensor { return nil }
			input2.SeedFunc = func() tensor.Tensor { return nil }

			weighted, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: 1,
			})
			if err != nil {
				t.Fatal(err)
			}

			ni1 := node.NewNode(input1)
			ni2 := node.NewNode(input2)
			nw := node.NewNode(weighted)

			ni1.AddChild(nw)
			ni2.AddChild(nw)
			nw.AddParent(ni1)
			nw.AddParent(ni2)

			ni1.SetNLayer(0)
			ni2.SetNLayer(0)
			nw.SetNLayer(1)

			err = nw.Forward()
			if err == nil {
				t.Fatal("expected error because of FC forward validation")
			} else if err.Error() != "FC input data validation failed: expected exactly one input tensor: got (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("FC node with nil result / Optimize(SGD) / returns error: gradient is nil", func(t *testing.T) {
			input1 := layers.NewInput()
			input2 := layers.NewInput()

			input1.SeedFunc = func() tensor.Tensor { return nil }
			input2.SeedFunc = func() tensor.Tensor { return nil }

			weighted, err := layers.NewFC(&layers.FCConfig{
				Inputs:  1,
				Outputs: 1,
			})
			if err != nil {
				t.Fatal(err)
			}

			optimizer, err := optimizers.NewSGD(nil)
			if err != nil {
				t.Fatal(err)
			}

			ni1 := node.NewNode(input1)
			ni2 := node.NewNode(input2)
			nw := node.NewNode(weighted)

			ni1.AddChild(nw)
			ni2.AddChild(nw)
			nw.AddParent(ni1)
			nw.AddParent(ni2)

			ni1.SetNLayer(0)
			ni2.SetNLayer(0)
			nw.SetNLayer(1)

			err = nw.Optimize(optimizer)
			if err == nil {
				t.Fatal("expected error because of optimizer validation")
			} else if err.Error() != "SGD input data validation failed: expected tensor's gradient not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

/* ----- helpers ----- */

type simple2DWeightedLayer struct {
	a tensor.Tensor
	b tensor.Tensor
}

func newSimpleWeightedLayer(conf *tensor.Config) (c *simple2DWeightedLayer, err error) {
	a, err := tensor.Full([]int{2}, 3., conf)
	if err != nil {
		return
	}

	b, err := tensor.Full([]int{2}, 2., conf)
	if err != nil {
		return
	}

	return &simple2DWeightedLayer{a: a, b: b}, nil
}

func (c *simple2DWeightedLayer) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	x := xs[0]

	y, err = x.Add(c.a)
	if err != nil {
		return
	}

	y, err = y.Add(c.b)
	if err != nil {
		return
	}

	return y, nil
}

func (c *simple2DWeightedLayer) Weights() []layers.Weight {
	return []layers.Weight{
		{Value: &c.a, Trainable: true},
		{Value: &c.b, Trainable: false},
	}
}
