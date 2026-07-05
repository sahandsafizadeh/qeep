package cputensor

import (
	"math"

	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

const float64EqualityThreshold = 1e-240

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

func (t *CPUTensor) dot(u *CPUTensor) *CPUTensor {
	t1, t2 := t, u
	dims := util.DotDims(t1.dims)

	o := new(CPUTensor)
	o.dims = dims
	o.strd = util.DimsToStrides(dims)
	o.data = make([]float64, util.DimsToNumElems(dims))

	nd := len(t1.dims)
	n := t1.dims[nd-1]
	t1s := t1.strd[nd-1]
	t2s := t2.strd[nd-1]

	kernel := func(t1ofst, t2ofsst, oofst int) {
		for k := range n {
			a := t1.data[t1ofst+k*t1s]
			b := t2.data[t2ofsst+k*t2s]
			o.data[oofst] += a * b
		}
	}

	bidx := make([]int, len(dims))
	nb := util.DimsToNumElems(dims)

	for range nb {
		var (
			t1ofst = 0
			t2ofst = 0
			oofst  = 0
		)
		for d := range dims {
			t1ofst += bidx[d] * t1.strd[d]
			t2ofst += bidx[d] * t2.strd[d]
			oofst += bidx[d] * o.strd[d]
		}

		kernel(t1ofst, t2ofst, oofst)
		updateElementWiseIndex(bidx, dims)
	}

	return o
}

func (t *CPUTensor) matMul(u *CPUTensor) *CPUTensor {
	t1, t2 := t, u
	dims := util.MatMulDims(t1.dims, t2.dims)

	o := new(CPUTensor)
	o.dims = dims
	o.strd = util.DimsToStrides(dims)
	o.data = make([]float64, util.DimsToNumElems(dims))

	nd := len(t1.dims)
	m := t1.dims[nd-2]
	n := t1.dims[nd-1]
	k := t2.dims[nd-1]
	t1rs := t1.strd[nd-2]
	t1cs := t1.strd[nd-1]
	t2rs := t2.strd[nd-2]
	t2cs := t2.strd[nd-1]
	ors := o.strd[nd-2]

	kernel := func(t1ofst, t2ofst, oofst int) {
		for i := range m {
			orow := oofst + i*ors
			t1row := t1ofst + i*t1rs
			for p := range n {
				t2row := t2ofst + p*t2rs
				a := t1.data[t1row+p*t1cs]
				for j := range k {
					b := t2.data[t2row+j*t2cs]
					o.data[orow+j] += a * b
				}
			}
		}
	}

	bdims := dims[:nd-2]
	bidx := make([]int, len(bdims))
	nb := util.DimsToNumElems(bdims)

	for range nb {
		var (
			t1ofst = 0
			t2ofst = 0
			oofst  = 0
		)
		for d := range bdims {
			t1ofst += bidx[d] * t1.strd[d]
			t2ofst += bidx[d] * t2.strd[d]
			oofst += bidx[d] * o.strd[d]
		}

		kernel(t1ofst, t2ofst, oofst)
		updateElementWiseIndex(bidx, bdims)
	}

	return o
}

func (t *CPUTensor) equals(u *CPUTensor) bool {
	o := t.eq(u)
	n := o.numElems()
	return o.sum() >= float64(n)
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
