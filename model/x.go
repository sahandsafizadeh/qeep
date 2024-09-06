package model

import (
	qt "github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

type networkNode struct {
	children []*networkNode
}

type Path struct {
	state *networkNode
}

func Input(shape ...int32) (p *Path) {
	return &Path{
		state: new(networkNode),
	}
}

func FC(n int32) func(*Path) *Path {
	return func(p *Path) *Path {
		n := new(networkNode) // of type fc
		p.state.children = append(p.state.children, n)

		r := &Path{
			state: n,
		}

		return r
	}
}

func Relu() func(*Path) *Path {
	return nil
}

func Tanh() func(*Path) *Path {
	return nil
}

func Sigmoid() func(*Path) *Path {
	return nil
}

func LeakyRelu(m float64) func(*Path) *Path {
	return nil
}

func testFunc() {
	x := Input(64)

	// first layer
	x = FC(32)(x)
	x = Relu()(x)

	// second layer
	x = FC(16)(x)
	x = Tanh()(x)

	// output layer
	x = FC(1)(x)
	x = Sigmoid()(x)

	_ = x
}

type Model struct {
	roots    []*networkNode
	lossFunc func(yp, yt qt.Tensor) (qt.Tensor, error)
}

func (c *Model) Fit(x qt.Tensor, y qt.Tensor) (err error) {
	// wrapped around batching and epoch mechanisms

	// err = bfsPass(c)
	if err != nil {
		return
	}

	l, err := c.lossFunc(nil, y) // todo: bfs pass must return a network output; it is actually the predict implementation
	if err != nil {
		return
	}

	err = tinit.BackProp(l)
	if err != nil {
		return
	}

	// update weights of network

	return nil
}

func (c *networkNode) Pass() {
}

// func bfsPass(m *Model) (err error) {
// 	q := new(queue)

// 	for _, n := range m.roots {
// 		q.push(n)
// 	}

// 	for !q.isEmpty() {
// 		var v *networkNode

// 		v, err = q.pop()
// 		if err != nil {
// 			return
// 		}

// 		v.Pass()

// 		for _, n := range v.children {
// 			q.push(n)
// 		}
// 	}

// 	return nil
// }

// path, flow, stream
