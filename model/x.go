package model

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
