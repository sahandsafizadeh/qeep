package cputensor

import (
	"math"

	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

const float64EqualityThreshold = 1e-12

func (t *CPUTensor) scale(u float64) *CPUTensor {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return u * a })
}

func (t *CPUTensor) pow(u float64) *CPUTensor {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Pow(a, u) })
}

func (t *CPUTensor) exp() *CPUTensor {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Exp(a) })
}

func (t *CPUTensor) log() *CPUTensor {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Log(a) })
}

func (t *CPUTensor) sin() *CPUTensor {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Sin(a) })
}

func (t *CPUTensor) cos() *CPUTensor {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Cos(a) })
}

func (t *CPUTensor) tan() *CPUTensor {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Tan(a) })
}

func (t *CPUTensor) sinh() *CPUTensor {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Sinh(a) })
}

func (t *CPUTensor) cosh() *CPUTensor {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Cosh(a) })
}

func (t *CPUTensor) tanh() *CPUTensor {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Tanh(a) })
}

func (t *CPUTensor) eq(u *CPUTensor) *CPUTensor {
	return applyBinaryFuncOnTensorsElemWise(t, u,
		func(a, b float64) float64 {
			if math.Abs(a-b) <= float64EqualityThreshold {
				return 1.
			} else {
				return 0.
			}
		})
}

func (t *CPUTensor) ne(u *CPUTensor) *CPUTensor {
	return applyBinaryFuncOnTensorsElemWise(t, u,
		func(a, b float64) float64 {
			if math.Abs(a-b) <= float64EqualityThreshold {
				return 0.
			} else {
				return 1.
			}
		})
}

func (t *CPUTensor) gt(u *CPUTensor) *CPUTensor {
	return applyBinaryFuncOnTensorsElemWise(t, u,
		func(a, b float64) float64 {
			if a > b {
				return 1.
			} else {
				return 0.
			}
		})
}

func (t *CPUTensor) ge(u *CPUTensor) *CPUTensor {
	return applyBinaryFuncOnTensorsElemWise(t, u,
		func(a, b float64) float64 {
			if a >= b {
				return 1.
			} else {
				return 0.
			}
		})
}

func (t *CPUTensor) lt(u *CPUTensor) *CPUTensor {
	return applyBinaryFuncOnTensorsElemWise(t, u,
		func(a, b float64) float64 {
			if a < b {
				return 1.
			} else {
				return 0.
			}
		})
}

func (t *CPUTensor) le(u *CPUTensor) *CPUTensor {
	return applyBinaryFuncOnTensorsElemWise(t, u,
		func(a, b float64) float64 {
			if a <= b {
				return 1.
			} else {
				return 0.
			}
		})
}

func (t *CPUTensor) elmax(u *CPUTensor) *CPUTensor {
	return applyBinaryFuncOnTensorsElemWise(t, u, func(a, b float64) float64 { return math.Max(a, b) })
}

func (t *CPUTensor) elmin(u *CPUTensor) *CPUTensor {
	return applyBinaryFuncOnTensorsElemWise(t, u, func(a, b float64) float64 { return math.Min(a, b) })
}

func (t *CPUTensor) add(u *CPUTensor) *CPUTensor {
	return applyBinaryFuncOnTensorsElemWise(t, u, func(a, b float64) float64 { return a + b })
}

func (t *CPUTensor) sub(u *CPUTensor) *CPUTensor {
	return applyBinaryFuncOnTensorsElemWise(t, u, func(a, b float64) float64 { return a - b })
}

func (t *CPUTensor) mul(u *CPUTensor) *CPUTensor {
	return applyBinaryFuncOnTensorsElemWise(t, u, func(a, b float64) float64 { return a * b })
}

func (t *CPUTensor) div(u *CPUTensor) *CPUTensor {
	return applyBinaryFuncOnTensorsElemWise(t, u, func(a, b float64) float64 { return a / b })
}

func applyUnaryFuncOnTensorElemWise(t *CPUTensor, suf scalarUnaryFunc) *CPUTensor {
	index := make([]int, len(t.dims))
	return newTensorWithElementWiseInit(t.dims, func() float64 {
		defer updateElementWiseIndex(index, t.dims)
		return suf(t.at(index))
	})
}

func applyBinaryFuncOnTensorsElemWise(t1, t2 *CPUTensor, sbf scalarBinaryFunc) *CPUTensor {
	index := make([]int, len(t1.dims))
	return newTensorWithElementWiseInit(t1.dims, func() float64 {
		defer updateElementWiseIndex(index, t1.dims)
		return sbf(t1.at(index), t2.at(index))
	})
}

func (t *CPUTensor) dot(u *CPUTensor) *CPUTensor {
	t1, t2 := t, u
	dims := util.DotDims(t1.dims)

	nd := len(t1.dims)
	n := t1.dims[nd-1]

	oidx := make([]int, len(dims))
	t1idx := make([]int, nd)
	t2idx := make([]int, nd)

	return newTensorWithElementWiseInit(dims, func() float64 {
		defer updateElementWiseIndex(oidx, dims)

		copy(t1idx, oidx)
		copy(t2idx, oidx)

		res := 0.
		for k := range n {
			t1idx[nd-1] = k
			t2idx[nd-1] = k

			res += t1.at(t1idx) * t2.at(t2idx)
		}

		return res
	})
}

func (t *CPUTensor) matMul(u *CPUTensor) *CPUTensor {
	t1, t2 := t, u
	dims := util.MatMulDims(t1.dims, t2.dims)

	nd := len(dims)
	n := t1.dims[nd-1] // shared dim (m×n × n×k = m×k)

	oidx := make([]int, nd)
	t1idx := make([]int, nd)
	t2idx := make([]int, nd)

	return newTensorWithElementWiseInit(dims, func() float64 {
		defer updateElementWiseIndex(oidx, dims)

		copy(t1idx, oidx)
		copy(t2idx, oidx)

		res := 0.
		for p := range n {
			t1idx[nd-2] = oidx[nd-2]
			t1idx[nd-1] = p
			t2idx[nd-2] = p
			t2idx[nd-1] = oidx[nd-1]

			res += t1.at(t1idx) * t2.at(t2idx)
		}

		return res
	})
}

func (t *CPUTensor) equals(u *CPUTensor) bool {
	o := t.eq(u)
	n := o.numElems()
	return o.sum() >= float64(n)
}
