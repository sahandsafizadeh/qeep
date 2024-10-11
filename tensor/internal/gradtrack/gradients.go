package gradtrack

import "github.com/sahandsafizadeh/qeep/tensor"

func Root() (gctx *GradContext) {
	return new(GradContext)
}

func Forbidden() (gctx *GradContext) {
	return &GradContext{trackForbidden: true}
}

func Concat(y tensor.Tensor, xs []tensor.Tensor, dim int32) (gctx *GradContext) {
	backEdges := make([]*backwardEdge, len(xs))

	var base int32
	for i := 0; i < len(backEdges); i++ {
		shape := xs[i].Shape()
		index := make([]tensor.Range, len(shape))

		index[dim] = tensor.Range{
			From: base,
			To:   base + shape[dim],
		}

		base = index[dim].To

		backEdges[i] = &backwardEdge{
			target: xs[i],
			gradFn: func() (tensor.Tensor, error) {
				return y.Gradient().Slice(index)
			},
		}
	}

	return &GradContext{backEdges: backEdges}
}

func Slice(y tensor.Tensor, x tensor.Tensor, index []tensor.Range) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					return toZeros(x).Patch(index, y.Gradient())
				},
			},
		},
	}
}

func Patch(y tensor.Tensor, x tensor.Tensor, p tensor.Tensor, index []tensor.Range) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Patch(index, toZeros(p))
				},
			},
			{
				target: p,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Slice(index)
				},
			},
		},
	}
}

func Transpose(y tensor.Tensor, x tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Transpose()
				},
			},
		},
	}
}

func Reshape(y tensor.Tensor, x tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Reshape(x.Shape())
				},
			},
		},
	}
}

func UnSqueeze(y tensor.Tensor, x tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Reshape(x.Shape())
				},
			},
		},
	}
}

func Squeeze(y tensor.Tensor, x tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Reshape(x.Shape())
				},
			},
		},
	}
}

func Flatten(y tensor.Tensor, x tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Reshape(x.Shape())
				},
			},
		},
	}
}

func Broadcast(y tensor.Tensor, x tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (o tensor.Tensor, err error) {
					gy := y.Gradient()
					srcDims := x.Shape()
					dstDims := y.Shape()
					lds := len(srcDims)
					ldd := len(dstDims)

					i := 0
					for ldd-i > lds {
						gy, err = gy.AvgAlong(0)
						if err != nil {
							return
						}

						i++
					}

					j := 0
					for i < ldd {
						if srcDims[j] != dstDims[i] {
							gy, err = gy.AvgAlong(int32(j))
							if err != nil {
								return
							}

							gy, err = gy.UnSqueeze(int32(j))
							if err != nil {
								return
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

func SumAlong(y tensor.Tensor, x tensor.Tensor, dim int32) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					return reducerBroadcasted(y.Gradient(), x, dim)
				},
			},
		},
	}
}

func MaxAlong(y tensor.Tensor, x tensor.Tensor, dim int32) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (o tensor.Tensor, err error) {
					gy, err := reducerBroadcasted(y.Gradient(), x, dim)
					if err != nil {
						return
					}

					yb, err := reducerBroadcasted(y, x, dim)
					if err != nil {
						return
					}

					gx, err := x.Eq(yb)
					if err != nil {
						return
					}

					return gy.Mul(gx)
				},
			},
		},
	}
}

func MinAlong(y tensor.Tensor, x tensor.Tensor, dim int32) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (o tensor.Tensor, err error) {
					gy, err := reducerBroadcasted(y.Gradient(), x, dim)
					if err != nil {
						return
					}

					yb, err := reducerBroadcasted(y, x, dim)
					if err != nil {
						return
					}

					gx, err := x.Eq(yb)
					if err != nil {
						return
					}

					return gy.Mul(gx)
				},
			},
		},
	}
}

func AvgAlong(y tensor.Tensor, x tensor.Tensor, dim int32) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (o tensor.Tensor, err error) {
					gy, err := reducerBroadcasted(y.Gradient(), x, dim)
					if err != nil {
						return
					}

					n := float64(x.Shape()[dim])

					return gy.Scale(1 / n), nil
				},
			},
		},
	}
}

func VarAlong(y tensor.Tensor, x tensor.Tensor, dim int32) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (o tensor.Tensor, err error) {
					gy, err := reducerBroadcasted(y.Gradient(), x, dim)
					if err != nil {
						return
					}

					n := x.Shape()[dim]
					if n == 1 {
						return toZeros(x), nil
					}

					u, err := x.MeanAlong(dim)
					if err != nil {
						return
					}

					u, err = u.UnSqueeze(dim)
					if err != nil {
						return
					}

					gx, err := x.Sub(u)
					if err != nil {
						return
					}

					gx = gx.Scale(2 / float64(n-1))

					return gy.Mul(gx)
				},
			},
		},
	}
}

func StdAlong(y tensor.Tensor, x tensor.Tensor, dim int32) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (o tensor.Tensor, err error) {
					gy, err := reducerBroadcasted(y.Gradient(), x, dim)
					if err != nil {
						return
					}

					n := x.Shape()[dim]
					if n == 1 {
						return toZeros(x), nil
					}

					u, err := x.MeanAlong(dim)
					if err != nil {
						return
					}

					u, err = u.UnSqueeze(dim)
					if err != nil {
						return
					}

					gx, err := x.Sub(u)
					if err != nil {
						return
					}

					y, err := y.UnSqueeze(dim)
					if err != nil {
						return
					}

					gx, err = gx.Div(y)
					if err != nil {
						return
					}

					gx = gx.Scale(1 / float64(n-1))

					return gy.Mul(gx)
				},
			},
		},
	}
}

func MeanAlong(y tensor.Tensor, x tensor.Tensor, dim int32) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (o tensor.Tensor, err error) {
					gy, err := reducerBroadcasted(y.Gradient(), x, dim)
					if err != nil {
						return
					}

					n := float64(x.Shape()[dim])

					return gy.Scale(1 / n), nil
				},
			},
		},
	}
}

func Scale(y tensor.Tensor, x tensor.Tensor, a float64) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Scale(a), nil
				},
			},
		},
	}
}

func Pow(y tensor.Tensor, x tensor.Tensor, a float64) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					gy := y.Gradient()
					gx := x.Pow(a - 1)
					gx = gx.Scale(a)

					return gy.Mul(gx)
				},
			},
		},
	}
}

func Exp(y tensor.Tensor, x tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Mul(y)
				},
			},
		},
	}
}

func Log(y tensor.Tensor, x tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Div(x)
				},
			},
		},
	}
}

func Sin(y tensor.Tensor, x tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					gy := y.Gradient()
					gx := x.Cos()

					return gy.Mul(gx)
				},
			},
		},
	}
}

func Cos(y tensor.Tensor, x tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					gy := y.Gradient()
					gx := x.Sin().Scale(-1)

					return gy.Mul(gx)
				},
			},
		},
	}
}

func Tan(y tensor.Tensor, x tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					gy := y.Gradient()
					gx := x.Cos().Pow(-2)

					return gy.Mul(gx)
				},
			},
		},
	}
}

func Sinh(y tensor.Tensor, x tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					gy := y.Gradient()
					gx := x.Cosh()

					return gy.Mul(gx)
				},
			},
		},
	}
}

func Cosh(y tensor.Tensor, x tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					gy := y.Gradient()
					gx := x.Sinh()

					return gy.Mul(gx)
				},
			},
		},
	}
}

func Tanh(y tensor.Tensor, x tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: x,
				gradFn: func() (tensor.Tensor, error) {
					gy := y.Gradient()
					gx := x.Cosh().Pow(-2)

					return gy.Mul(gx)
				},
			},
		},
	}
}

func ElMax(y tensor.Tensor, a tensor.Tensor, b tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (o tensor.Tensor, err error) {
					gy := y.Gradient()

					ga, err := y.Eq(a)
					if err != nil {
						return
					}

					eq, err := a.Eq(b)
					if err != nil {
						return
					}

					ga, err = ga.Sub(eq.Scale(0.5))
					if err != nil {
						return
					}

					return gy.Mul(ga)
				},
			},
			{
				target: b,
				gradFn: func() (o tensor.Tensor, err error) {
					gy := y.Gradient()

					gb, err := y.Eq(b)
					if err != nil {
						return
					}

					eq, err := b.Eq(a)
					if err != nil {
						return
					}

					gb, err = gb.Sub(eq.Scale(0.5))
					if err != nil {
						return
					}

					return gy.Mul(gb)
				},
			},
		},
	}
}

func ElMin(y tensor.Tensor, a tensor.Tensor, b tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (o tensor.Tensor, err error) {
					gy := y.Gradient()

					ga, err := y.Eq(a)
					if err != nil {
						return
					}

					eq, err := a.Eq(b)
					if err != nil {
						return
					}

					ga, err = ga.Sub(eq.Scale(0.5))
					if err != nil {
						return
					}

					return gy.Mul(ga)
				},
			},
			{
				target: b,
				gradFn: func() (o tensor.Tensor, err error) {
					gy := y.Gradient()

					gb, err := y.Eq(b)
					if err != nil {
						return
					}

					eq, err := b.Eq(a)
					if err != nil {
						return
					}

					gb, err = gb.Sub(eq.Scale(0.5))
					if err != nil {
						return
					}

					return gy.Mul(gb)
				},
			},
		},
	}
}

func Add(y tensor.Tensor, a tensor.Tensor, b tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient(), nil
				},
			},
			{
				target: b,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient(), nil
				},
			},
		},
	}
}

func Sub(y tensor.Tensor, a tensor.Tensor, b tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient(), nil
				},
			},
			{
				target: b,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Scale(-1), nil
				},
			},
		},
	}
}

func Mul(y tensor.Tensor, a tensor.Tensor, b tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Mul(b)
				},
			},
			{
				target: b,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Mul(a)
				},
			},
		},
	}
}

func Div(y tensor.Tensor, a tensor.Tensor, b tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Div(b)
				},
			},
			{
				target: b,
				gradFn: func() (o tensor.Tensor, err error) {
					gy := y.Gradient()

					n := a.Scale(-1)
					d := b.Pow(2)

					gb, err := n.Div(d)
					if err != nil {
						return
					}

					return gy.Mul(gb)
				},
			},
		},
	}
}

func Dot(y tensor.Tensor, a tensor.Tensor, b tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Mul(b)
				},
			},
			{
				target: b,
				gradFn: func() (tensor.Tensor, error) {
					return y.Gradient().Mul(a)
				},
			},
		},
	}
}

func MatMul(y tensor.Tensor, a tensor.Tensor, b tensor.Tensor) (gctx *GradContext) {
	return &GradContext{
		backEdges: []*backwardEdge{
			{
				target: a,
				gradFn: func() (o tensor.Tensor, err error) {
					gy := y.Gradient()

					ga, err := b.Transpose()
					if err != nil {
						return
					}

					return gy.MatMul(ga)
				},
			},
			{
				target: b,
				gradFn: func() (o tensor.Tensor, err error) {
					gy := y.Gradient()

					gb, err := a.Transpose()
					if err != nil {
						return
					}

					return gb.MatMul(gy)
				},
			},
		},
	}
}
