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

func TestNodeInitialization(t *testing.T) {

	/* ------------------------------ */

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

	/* ------------------------------ */

	if np1.Layer() != lp1 {
		t.Fatalf("unexpected node Layer")
	}

	if np2.Layer() != lp2 {
		t.Fatalf("unexpected node Layer")
	}

	if np3.Layer() != lp3 {
		t.Fatalf("unexpected node Layer")
	}

	if nm1.Layer() != lm1 {
		t.Fatalf("unexpected node Layer")
	}

	if nc1.Layer() != lc1 {
		t.Fatalf("unexpected node Layer")
	}

	if nc2.Layer() != lc2 {
		t.Fatalf("unexpected node Layer")
	}

	/* ------------------------------ */

	if np1.Result() != nil {
		t.Fatalf("unexpected node Result")
	}

	if np2.Result() != nil {
		t.Fatalf("unexpected node Result")
	}

	if np3.Result() != nil {
		t.Fatalf("unexpected node Result")
	}

	if nm1.Result() != nil {
		t.Fatalf("unexpected node Result")
	}

	if nc1.Result() != nil {
		t.Fatalf("unexpected node Result")
	}

	if nc2.Result() != nil {
		t.Fatalf("unexpected node Result")
	}

	/* ------------------------------ */

	if !slices.Equal(np1.Parents(), nil) {
		t.Fatalf("unexpected node Parents")
	}

	if !slices.Equal(np2.Parents(), nil) {
		t.Fatalf("unexpected node Parents")
	}

	if !slices.Equal(np3.Parents(), nil) {
		t.Fatalf("unexpected node Parents")
	}

	if !slices.Equal(nm1.Parents(), []*node.Node{np1, np2, np3}) {
		t.Fatalf("unexpected node Parents")
	}

	if !slices.Equal(nc1.Parents(), []*node.Node{nm1}) {
		t.Fatalf("unexpected node Parents")
	}

	if !slices.Equal(nc2.Parents(), []*node.Node{nm1}) {
		t.Fatalf("unexpected node Parents")
	}

	/* ------------------------------ */

	if !slices.Equal(np1.Children(), []*node.Node{nm1}) {
		t.Fatalf("unexpected node Children")
	}

	if !slices.Equal(np2.Children(), []*node.Node{nm1}) {
		t.Fatalf("unexpected node Children")
	}

	if !slices.Equal(np3.Children(), []*node.Node{nm1}) {
		t.Fatalf("unexpected node Children")
	}

	if !slices.Equal(nm1.Children(), []*node.Node{nc1, nc2}) {
		t.Fatalf("unexpected node Children")
	}

	if !slices.Equal(nc1.Children(), nil) {
		t.Fatalf("unexpected node Children")
	}

	if !slices.Equal(nc2.Children(), nil) {
		t.Fatalf("unexpected node Children")
	}

	/* ------------------------------ */

}

func TestNodeOperation(t *testing.T) {
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

		input := layers.NewInput()
		input.SeedFunc = func() tensor.Tensor {
			x, err := tensor.Full([]int{2}, 1., confU)
			if err != nil {
				t.Fatal(err)
			}

			return x
		}

		weighted, err := newSimpleWeightedLayer(confT)
		if err != nil {
			t.Fatal(err)
		}

		activation := activations.NewRelu()

		optimizer := optimizers.NewSGD(&optimizers.SGDConfig{LearningRate: 1.})

		/* ------------------------------ */

		ni := node.NewNode(input)
		nw := node.NewNode(weighted)
		na := node.NewNode(activation)

		ni.AddChild(nw)
		nw.AddParent(ni)
		nw.AddChild(na)
		na.AddParent(nw)

		/* ------------------------------ */

		ni.EnableGrad()
		nw.EnableGrad()
		na.EnableGrad()

		/* --------------- */

		err = ni.Forward()
		if err != nil {
			t.Fatal(err)
		}

		err = nw.Forward()
		if err != nil {
			t.Fatal(err)
		}

		err = na.Forward()
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act := ni.Result()

		exp, err := tensor.Full([]int{2}, 1., confU)
		if err != nil {
			return
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		act = nw.Result()

		exp, err = tensor.Full([]int{2}, 6., confU)
		if err != nil {
			return
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		act = na.Result()

		exp, err = tensor.Full([]int{2}, 6., confU)
		if err != nil {
			return
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		err = tensor.BackPropagate(na.Result())
		if err != nil {
			t.Fatal(err)
		}

		err = ni.Optimize(optimizer)
		if err != nil {
			t.Fatal(err)
		}

		err = nw.Optimize(optimizer)
		if err != nil {
			t.Fatal(err)
		}

		err = na.Optimize(optimizer)
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act = weighted.a

		exp, err = tensor.Full([]int{2}, 2., confT)
		if err != nil {
			return
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		act = weighted.b

		exp, err = tensor.Full([]int{2}, 2., confT)
		if err != nil {
			return
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

		ni.DisableGrad()
		nw.DisableGrad()
		na.DisableGrad()

		/* --------------- */

		err = ni.Forward()
		if err != nil {
			t.Fatal(err)
		}

		err = nw.Forward()
		if err != nil {
			t.Fatal(err)
		}

		err = na.Forward()
		if err != nil {
			t.Fatal(err)
		}

		/* --------------- */

		act = ni.Result()

		exp, err = tensor.Full([]int{2}, 1., confU)
		if err != nil {
			return
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		act = nw.Result()

		exp, err = tensor.Full([]int{2}, 5., confU)
		if err != nil {
			return
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		act = na.Result()

		exp, err = tensor.Full([]int{2}, 5., confU)
		if err != nil {
			return
		}

		if eq, err := act.Equals(exp); err != nil {
			t.Fatal(err)
		} else if !eq {
			t.Fatalf("expected tensors to be equal")
		}

		/* ------------------------------ */

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