package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor/internal/core"

func Transfer(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					to := x.Device()
					gy := y.Gradient()
					gy = gy.(core.ExporterTensor)

					return transferFunc(gy, to)
				},
			},
		},
	}
}

func Concat(y core.Tensor, xs []core.Tensor, dim int) *GradContext {
	if anyIsBPDirty(xs...) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(xs...) {
		return NewGradContext(false)
	}

	backEdges := make([]*backwardEdge, len(xs))

	var base int
	for i := range backEdges {
		shape := xs[i].Shape()
		index := make([]core.Range, len(shape))

		index[dim] = core.Range{
			From: base,
			To:   base + shape[dim],
		}

		base = index[dim].To

		backEdges[i] = &backwardEdge{
			target: xs[i],
			gradFn: func() (core.Tensor, error) {
				return y.Gradient().Slice(index)
			},
		}
	}

	return &GradContext{
		tracked:   true,
		backEdges: backEdges,
	}
}

func Slice(y core.Tensor, x core.Tensor, index []core.Range) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					return toZeros(x).Patch(index, y.Gradient())
				},
			},
		},
	}
}

func Transpose(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Transpose()
				},
			},
		},
	}
}

func Reshape(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Reshape(x.Shape())
				},
			},
		},
	}
}

func Flatten(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Reshape(x.Shape())
				},
			},
		},
	}
}

func UnSqueeze(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Reshape(x.Shape())
				},
			},
		},
	}
}

func Squeeze(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Reshape(x.Shape())
				},
			},
		},
	}
}

func Broadcast(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (o core.Tensor, err error) {
					gy := y.Gradient()
					srcDims := x.Shape()
					dstDims := y.Shape()
					lds := len(srcDims)
					ldd := len(dstDims)

					i := 0
					for ldd-i > lds {
						gy, err = gy.SumAlong(0)
						if err != nil {
							return o, err
						}

						i++
					}

					j := 0
					for i < ldd {
						if srcDims[j] != dstDims[i] {
							gy, err = gy.SumAlong(j)
							if err != nil {
								return o, err
							}

							gy, err = gy.UnSqueeze(j)
							if err != nil {
								return o, err
							}
						}

						j++
						i++
					}

					return gy, nil
				},
			},
		},
	}
}

func SumAlong(y core.Tensor, x core.Tensor, dim int) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					return reducerBroadcasted(y.Gradient(), x, dim)
				},
			},
		},
	}
}

func MaxAlong(y core.Tensor, x core.Tensor, dim int) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (o core.Tensor, err error) {
					gy, err := reducerBroadcasted(y.Gradient(), x, dim)
					if err != nil {
						return o, err
					}

					yb, err := reducerBroadcasted(y, x, dim)
					if err != nil {
						return o, err
					}

					gx, err := x.Eq(yb)
					if err != nil {
						return o, err
					}

					cnt, err := gx.SumAlong(dim)
					if err != nil {
						return o, err
					}

					cnt, err = reducerBroadcasted(cnt, x, dim)
					if err != nil {
						return o, err
					}

					gx, err = gx.Div(cnt)
					if err != nil {
						return o, err
					}

					return gy.Mul(gx)
				},
			},
		},
	}
}

func MinAlong(y core.Tensor, x core.Tensor, dim int) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (o core.Tensor, err error) {
					gy, err := reducerBroadcasted(y.Gradient(), x, dim)
					if err != nil {
						return o, err
					}

					yb, err := reducerBroadcasted(y, x, dim)
					if err != nil {
						return o, err
					}

					gx, err := x.Eq(yb)
					if err != nil {
						return o, err
					}

					cnt, err := gx.SumAlong(dim)
					if err != nil {
						return o, err
					}

					cnt, err = reducerBroadcasted(cnt, x, dim)
					if err != nil {
						return o, err
					}

					gx, err = gx.Div(cnt)
					if err != nil {
						return o, err
					}

					return gy.Mul(gx)
				},
			},
		},
	}
}

func AvgAlong(y core.Tensor, x core.Tensor, dim int) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (o core.Tensor, err error) {
					gy, err := reducerBroadcasted(y.Gradient(), x, dim)
					if err != nil {
						return o, err
					}

					n := float64(x.Shape()[dim])

					return gy.Scale(1 / n), nil
				},
			},
		},
	}
}

func VarAlong(y core.Tensor, x core.Tensor, dim int) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (o core.Tensor, err error) {
					gy, err := reducerBroadcasted(y.Gradient(), x, dim)
					if err != nil {
						return o, err
					}

					n := x.Shape()[dim]
					if n == 1 {
						return toZeros(x), nil
					}

					u, err := x.MeanAlong(dim)
					if err != nil {
						return o, err
					}

					u, err = u.UnSqueeze(dim)
					if err != nil {
						return o, err
					}

					gx, err := x.Sub(u)
					if err != nil {
						return o, err
					}

					gx = gx.Scale(2 / float64(n-1))

					return gy.Mul(gx)
				},
			},
		},
	}
}

func StdAlong(y core.Tensor, x core.Tensor, dim int) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (o core.Tensor, err error) {
					gy, err := reducerBroadcasted(y.Gradient(), x, dim)
					if err != nil {
						return o, err
					}

					n := x.Shape()[dim]
					if n == 1 {
						return toZeros(x), nil
					}

					u, err := x.MeanAlong(dim)
					if err != nil {
						return o, err
					}

					u, err = u.UnSqueeze(dim)
					if err != nil {
						return o, err
					}

					gx, err := x.Sub(u)
					if err != nil {
						return o, err
					}

					y, err := y.UnSqueeze(dim)
					if err != nil {
						return o, err
					}

					gx, err = gx.Div(y)
					if err != nil {
						return o, err
					}

					gx = gx.Scale(1 / float64(n-1))

					return gy.Mul(gx)
				},
			},
		},
	}
}

func MeanAlong(y core.Tensor, x core.Tensor, dim int) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (o core.Tensor, err error) {
					gy, err := reducerBroadcasted(y.Gradient(), x, dim)
					if err != nil {
						return o, err
					}

					n := float64(x.Shape()[dim])

					return gy.Scale(1 / n), nil
				},
			},
		},
	}
}

func Scale(y core.Tensor, x core.Tensor, a float64) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Scale(a), nil
				},
			},
		},
	}
}

func Pow(y core.Tensor, x core.Tensor, a float64) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					gy := y.Gradient()
					gx := x.Pow(a - 1)
					gx = gx.Scale(a)

					return gy.Mul(gx)
				},
			},
		},
	}
}

func Exp(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Mul(y)
				},
			},
		},
	}
}

func Log(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Div(x)
				},
			},
		},
	}
}

func Sin(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					gy := y.Gradient()
					gx := x.Cos()

					return gy.Mul(gx)
				},
			},
		},
	}
}

func Cos(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					gy := y.Gradient()
					gx := x.Sin().Scale(-1)

					return gy.Mul(gx)
				},
			},
		},
	}
}

func Tan(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					gy := y.Gradient()
					gx := x.Cos().Pow(-2)

					return gy.Mul(gx)
				},
			},
		},
	}
}

func Sinh(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					gy := y.Gradient()
					gx := x.Cosh()

					return gy.Mul(gx)
				},
			},
		},
	}
}

func Cosh(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					gy := y.Gradient()
					gx := x.Sinh()

					return gy.Mul(gx)
				},
			},
		},
	}
}

func Tanh(y core.Tensor, x core.Tensor) *GradContext {
	if anyIsBPDirty(x) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					gy := y.Gradient()
					gx := x.Cosh().Pow(-2)

					return gy.Mul(gx)
				},
			},
		},
	}
}

func ElMax(y core.Tensor, a core.Tensor, b core.Tensor) *GradContext {
	if anyIsBPDirty(a, b) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(a, b) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (o core.Tensor, err error) {
					gy := y.Gradient()

					ga, err := y.Eq(a)
					if err != nil {
						return o, err
					}

					eq, err := a.Eq(b)
					if err != nil {
						return o, err
					}

					ga, err = ga.Sub(eq.Scale(0.5))
					if err != nil {
						return o, err
					}

					return gy.Mul(ga)
				},
			},
			{
				target: b,
				gradFn: func() (o core.Tensor, err error) {
					gy := y.Gradient()

					gb, err := y.Eq(b)
					if err != nil {
						return o, err
					}

					eq, err := b.Eq(a)
					if err != nil {
						return o, err
					}

					gb, err = gb.Sub(eq.Scale(0.5))
					if err != nil {
						return o, err
					}

					return gy.Mul(gb)
				},
			},
		},
	}
}

func ElMin(y core.Tensor, a core.Tensor, b core.Tensor) *GradContext {
	if anyIsBPDirty(a, b) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(a, b) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (o core.Tensor, err error) {
					gy := y.Gradient()

					ga, err := y.Eq(a)
					if err != nil {
						return o, err
					}

					eq, err := a.Eq(b)
					if err != nil {
						return o, err
					}

					ga, err = ga.Sub(eq.Scale(0.5))
					if err != nil {
						return o, err
					}

					return gy.Mul(ga)
				},
			},
			{
				target: b,
				gradFn: func() (o core.Tensor, err error) {
					gy := y.Gradient()

					gb, err := y.Eq(b)
					if err != nil {
						return o, err
					}

					eq, err := b.Eq(a)
					if err != nil {
						return o, err
					}

					gb, err = gb.Sub(eq.Scale(0.5))
					if err != nil {
						return o, err
					}

					return gy.Mul(gb)
				},
			},
		},
	}
}

func Add(y core.Tensor, a core.Tensor, b core.Tensor) *GradContext {
	if anyIsBPDirty(a, b) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(a, b) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient(), nil
				},
			},
			{
				target: b,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient(), nil
				},
			},
		},
	}
}

func Sub(y core.Tensor, a core.Tensor, b core.Tensor) *GradContext {
	if anyIsBPDirty(a, b) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(a, b) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient(), nil
				},
			},
			{
				target: b,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Scale(-1), nil
				},
			},
		},
	}
}

func Mul(y core.Tensor, a core.Tensor, b core.Tensor) *GradContext {
	if anyIsBPDirty(a, b) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(a, b) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Mul(b)
				},
			},
			{
				target: b,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Mul(a)
				},
			},
		},
	}
}

func Div(y core.Tensor, a core.Tensor, b core.Tensor) *GradContext {
	if anyIsBPDirty(a, b) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(a, b) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Div(b)
				},
			},
			{
				target: b,
				gradFn: func() (o core.Tensor, err error) {
					gy := y.Gradient()

					gb, err := y.Scale(-1).Div(b)
					if err != nil {
						return o, err
					}

					return gy.Mul(gb)
				},
			},
		},
	}
}

func Dot(y core.Tensor, a core.Tensor, b core.Tensor) *GradContext {
	if anyIsBPDirty(a, b) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(a, b) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Mul(b)
				},
			},
			{
				target: b,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Mul(a)
				},
			},
		},
	}
}

func MatMul(y core.Tensor, a core.Tensor, b core.Tensor) *GradContext {
	if anyIsBPDirty(a, b) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(a, b) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (o core.Tensor, err error) {
					gy := y.Gradient()

					ga, err := b.Transpose()
					if err != nil {
						return o, err
					}

					return gy.MatMul(ga)
				},
			},
			{
				target: b,
				gradFn: func() (o core.Tensor, err error) {
					gy := y.Gradient()

					gb, err := a.Transpose()
					if err != nil {
						return o, err
					}

					return gb.MatMul(gy)
				},
			},
		},
	}
}

func Patch(y core.Tensor, x core.Tensor, p core.Tensor, index []core.Range) *GradContext {
	if anyIsBPDirty(x, p) {
		return NewDirtyGradContext()
	}
	if nonIsTracked(x, p) {
		return NewGradContext(false)
	}

	return &GradContext{
		tracked: true,
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Patch(index, toZeros(p))
				},
			},
			{
				target: p,
				gradFn: func() (core.Tensor, error) {
					return y.Gradient().Slice(index)
				},
			},
		},
	}
}
