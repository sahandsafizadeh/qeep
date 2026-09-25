package tensor_test

import (
	"math"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestBackPropagate(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main functionalities ==============================

		t.Run("chain of operations | BackPropagate | gradient equals product of all scales", func(t *testing.T) {
			a, err := tensor.Full([]int{3, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := a.Scale(2.).Scale(3.).Scale(5.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := a.Gradient()

			h, err := tensor.Full([]int{3, 2}, 30., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("diamond DAG: tracked tensor fans out to Sin and Cos then adds | BackPropagate | gradient equals cos(a) minus sin(a)", func(t *testing.T) {
			a, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			s := a.Sin()
			c := a.Cos()

			y, err := s.Add(c)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := a.Gradient()

			h, err := tensor.Full(nil, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("tensor used in multiple branches | BackPropagate | gradient accumulates from all branches", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			x1 := a.Scale(2.)
			x2 := a.Scale(3.)
			x3 := a.Scale(5.)

			y, err := tensor.Concat([]tensor.Tensor{x1, x2, x3}, 0)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := a.Gradient()

			h, err := tensor.Full([]int{2, 3}, 10., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("shared leaf fed through Add edges | BackPropagate | gradient accumulation not corrupted", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			s, err := a.Add(b)
			if err != nil {
				t.Fatal(err)
			}
			y, err := s.Add(a)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			ga := a.Gradient()
			gb := b.Gradient()

			ha, err := tensor.Full([]int{2, 2}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			hb, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := ga.Equals(ha); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := gb.Equals(hb); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("residual (skip) connection: f(x) + x where x is a non-leaf | BackPropagate | gradient of x finalized before propagating to leaf", func(t *testing.T) {
			a, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			x := a.Scale(2.)
			f1 := x.Scale(3.)
			f2 := f1.Scale(4.)

			y, err := f2.Add(x)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := a.Gradient()

			h, err := tensor.Full(nil, 26., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("element-wise value-dependent residual over a non-leaf tensor | BackPropagate | per-element gradient finalized before propagating", func(t *testing.T) {
			a, err := tensor.Of([][]float64{
				{1., 2.},
				{3., 4.},
			}, &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			x := a.Pow(2.)
			f1 := x.Scale(3.)
			f2 := f1.Scale(5.)

			y, err := f2.Add(x)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := a.Gradient()

			h, err := tensor.Of([][]float64{
				{32., 64.},
				{96., 128.},
			}, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("shared non-leaf with a long pending chain | BackPropagate | gradient not propagated until deep contributions arrive", func(t *testing.T) {
			a, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			c := a.Scale(2.)
			s1 := c.Scale(3.)
			s2 := s1.Scale(4.)
			s3 := s2.Scale(5.)

			y, err := s3.Add(c)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := a.Gradient()

			h, err := tensor.Full(nil, 122., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("diamond DAG through a non-leaf intermediate | BackPropagate | gradient is finalized before propagating upstream (no double-counting)", func(t *testing.T) {
			a, err := tensor.Full(nil, 0., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			m := a.Scale(2.)
			s := m.Sin()
			c := m.Cos()

			y, err := s.Add(c)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := a.Gradient()

			h, err := tensor.Full(nil, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("value-dependent Mul diamond with shared non-leaf dequeued before its sibling contributes | BackPropagate | gradient correct", func(t *testing.T) {
			a, err := tensor.Full(nil, 2., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			c := a.Scale(3.)
			e := c.Scale(5.)

			// c is the FIRST edge; a simple BFS would process c before the long path of e to c.
			y, err := c.Mul(e)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := a.Gradient()

			h, err := tensor.Full(nil, 180., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("nested residuals: two stacked convergence points at different depths | BackPropagate | gradient accumulates correctly through both", func(t *testing.T) {
			a, err := tensor.Full(nil, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			b := a.Scale(2.)
			p1 := b.Scale(3.)
			p2 := p1.Scale(5.)

			s1, err := p2.Add(b)
			if err != nil {
				t.Fatal(err)
			}

			y, err := s1.Add(p1)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := a.Gradient()

			h, err := tensor.Full(nil, 38., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== partial propagation ==============================

		t.Run("multiple independent outputs from same tracked leaf | BackPropagate on one then the other | gradients accumulate without reset", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			a := x.Scale(2.)
			b := x.Scale(3.)

			err = tensor.BackPropagate(a)
			if err != nil {
				t.Fatal(err)
			}

			g := x.Gradient()

			h, err := tensor.Full([]int{2, 2}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			err = tensor.BackPropagate(b)
			if err != nil {
				t.Fatal(err)
			}

			g = x.Gradient()

			h, err = tensor.Full([]int{2, 2}, 5., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("multiple outputs sharing a non-leaf ancestor | BackPropagate on one then the other | gradients of non-leaf and leaf accumulate across runs", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			m := x.Scale(2.)
			a := m.Scale(3.)
			b := m.Scale(5.)

			err = tensor.BackPropagate(a)
			if err != nil {
				t.Fatal(err)
			}

			gm := m.Gradient()
			gx := x.Gradient()

			hm, err := tensor.Full([]int{2, 2}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			hx, err := tensor.Full([]int{2, 2}, 6., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := gm.Equals(hm); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := gx.Equals(hx); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			err = tensor.BackPropagate(b)
			if err != nil {
				t.Fatal(err)
			}

			gm = m.Gradient()
			gx = x.Gradient()

			hm, err = tensor.Full([]int{2, 2}, 8., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			hx, err = tensor.Full([]int{2, 2}, 16., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := gm.Equals(hm); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := gx.Equals(hx); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("root of the graph | BackPropagate twice without reset | gradient of the root accumulates like any other node", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := a.Scale(2.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := y.Gradient()

			h, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g = y.Gradient()

			h, err = tensor.Full([]int{2, 2}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("non-leaf accumulates gradient as a branch target then is used as its own root | BackPropagate on the branch then on the node itself | gradient combines both contributions", func(t *testing.T) {
			x, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			m := x.Scale(2.)
			a := m.Scale(3.)

			err = tensor.BackPropagate(a)
			if err != nil {
				t.Fatal(err)
			}

			gm := m.Gradient()
			gx := x.Gradient()

			hm, err := tensor.Full([]int{2, 2}, 3., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			hx, err := tensor.Full([]int{2, 2}, 6., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := gm.Equals(hm); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := gx.Equals(hx); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			err = tensor.BackPropagate(m)
			if err != nil {
				t.Fatal(err)
			}

			gm = m.Gradient()
			gx = x.Gradient()

			hm, err = tensor.Full([]int{2, 2}, 4., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			hx, err = tensor.Full([]int{2, 2}, 8., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := gm.Equals(hm); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := gx.Equals(hx); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("two independent chains from unrelated leaves | BackPropagate on one then the other | gradients of unrelated leaves remain isolated", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y1 := a.Scale(2.).Scale(3.)
			y2 := b.Scale(4.).Scale(5.)

			err = tensor.BackPropagate(y1)
			if err != nil {
				t.Fatal(err)
			}

			ga := a.Gradient()

			ha, err := tensor.Full([]int{2, 2}, 6., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := ga.Equals(ha); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if b.Gradient() != nil {
				t.Fatal("expected gradient to be nil")
			}

			err = tensor.BackPropagate(y2)
			if err != nil {
				t.Fatal(err)
			}

			ga = a.Gradient()
			gb := b.Gradient()

			hb, err := tensor.Full([]int{2, 2}, 20., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := gb.Equals(hb); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := ga.Equals(ha); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("same output | BackPropagate twice without reset | gradient doubles", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y := a.Scale(2.).Scale(3.)

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := a.Gradient()

			h, err := tensor.Full([]int{2, 2}, 6., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g = a.Gradient()

			h, err = tensor.Full([]int{2, 2}, 12., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== tracking/resetting ==============================

		t.Run("mixed tracked and untracked computation graph | BackPropagate | gradient set on tracked nodes, nil on untracked", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 3}, 3., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full([]int{2, 3}, 2., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}
			c, err := tensor.Full([]int{2, 3}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: false,
			})
			if err != nil {
				t.Fatal(err)
			}

			x1 := a.Scale(2.)
			x2 := b.Scale(3.)
			x3 := c.Scale(5.)

			t1, err := x1.Add(x2)
			if err != nil {
				t.Fatal(err)
			}
			t2, err := x3.Add(x2)
			if err != nil {
				t.Fatal(err)
			}

			y, err := tensor.Concat([]tensor.Tensor{t1, t2}, 0)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			if a.Gradient() == nil {
				t.Fatal("expected a gradient not to be nil")
			}
			if b.Gradient() != nil {
				t.Fatal("expected b gradient to be nil")
			}
			if c.Gradient() != nil {
				t.Fatal("expected c gradient to be nil")
			}
			if x1.Gradient() == nil {
				t.Fatal("expected x1 gradient not to be nil")
			}
			if x2.Gradient() != nil {
				t.Fatal("expected x2 gradient to be nil")
			}
			if x3.Gradient() != nil {
				t.Fatal("expected x3 gradient to be nil")
			}
			if t1.Gradient() == nil {
				t.Fatal("expected t1 gradient not to be nil")
			}
			if t2.Gradient() != nil {
				t.Fatal("expected t2 gradient to be nil")
			}
			if y.Gradient() == nil {
				t.Fatal("expected y gradient not to be nil")
			}
		})

		t.Run("tracked leaf after BackPropagate and ResetGradient | BackPropagate on a rebuilt identical Scale chain | gradient equals a single fresh run", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			y1 := a.Scale(2.).Scale(3.)

			err = tensor.BackPropagate(y1)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.ResetGradient(a, true)
			if err != nil {
				t.Fatal(err)
			}

			y2 := a.Scale(2.).Scale(3.)

			err = tensor.BackPropagate(y2)
			if err != nil {
				t.Fatal(err)
			}

			g := a.Gradient()

			h, err := tensor.Full([]int{2, 2}, 6., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("shared-leaf Add graph after BackPropagate and ResetGradient on both leaves | BackPropagate on a rebuilt identical graph | gradient matches a single fresh run", func(t *testing.T) {
			a, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			b, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			s1, err := a.Add(b)
			if err != nil {
				t.Fatal(err)
			}
			y1, err := s1.Add(a)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y1)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.ResetGradient(a, true)
			if err != nil {
				t.Fatal(err)
			}
			err = tensor.ResetGradient(b, true)
			if err != nil {
				t.Fatal(err)
			}

			s2, err := a.Add(b)
			if err != nil {
				t.Fatal(err)
			}
			y2, err := s2.Add(a)
			if err != nil {
				t.Fatal(err)
			}

			err = tensor.BackPropagate(y2)
			if err != nil {
				t.Fatal(err)
			}

			ga := a.Gradient()
			gb := b.Gradient()

			ha, err := tensor.Full([]int{2, 2}, 2., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			hb, err := tensor.Full([]int{2, 2}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := ga.Equals(ha); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
			if eq, err := gb.Equals(hb); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== extra functionalities ==============================

		t.Run("medium chain of Linear+bias+residual blocks over a wide 2^20-element tensor | BackPropagate | gradient equals product of per-block (weight+1) factors", func(t *testing.T) {
			n := 1 << 20

			a, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			ws := []float64{2., 3., 2., 3., 2., 3., 2., 3.}

			b, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			p := a
			for _, w := range ws {
				x := p.Scale(w)

				x, err = x.Add(b)
				if err != nil {
					t.Fatal(err)
				}

				p, err = p.Add(x)
				if err != nil {
					t.Fatal(err)
				}
			}
			y := p

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := a.Gradient()

			h, err := tensor.Full([]int{1, n}, 20736., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		t.Run("very long chain of a shared-weight Linear+bias+residual block unrolled 100 times over a 2^10-element tensor | BackPropagate | gradient equals (weight+1) raised to the chain length", func(t *testing.T) {
			const (
				n  = 1 << 10
				ni = 100
			)

			a, err := tensor.Full([]int{1, n}, 1., &tensor.Config{
				Device:    dev,
				GradTrack: true,
			})
			if err != nil {
				t.Fatal(err)
			}

			w := 1.

			b, err := tensor.Full([]int{1, n}, 1., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			p := a
			for range ni {
				x := p.Scale(w)

				x, err := x.Add(b)
				if err != nil {
					t.Fatal(err)
				}

				p, err = p.Add(x)
				if err != nil {
					t.Fatal(err)
				}
			}
			y := p

			err = tensor.BackPropagate(y)
			if err != nil {
				t.Fatal(err)
			}

			g := a.Gradient()

			h, err := tensor.Full([]int{1, n}, math.Pow(w+1, ni), &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if eq, err := g.Equals(h); err != nil {
				t.Fatal(err)
			} else if !eq {
				t.Fatal("expected tensors to be equal")
			}
		})

		// ============================== validations ==============================

		t.Run("nil tensor | ResetGradient | returns error: unsupported tensor implementation", func(t *testing.T) {
			err := tensor.ResetGradient(nil, true)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != "ResetGradient tensor implementation validation failed: unsupported tensor implementation" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("nil tensor | BackPropagate | returns error: unsupported tensor implementation", func(t *testing.T) {
			err := tensor.BackPropagate(nil)
			if err == nil {
				t.Fatal("expected error because of nil input tensor")
			} else if err.Error() != "BackPropagate tensor implementation validation failed: unsupported tensor implementation" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}
