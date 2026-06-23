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
	nd := len(t1.dims)
	n := t1.dims[nd-1]

	t1Stride := t1.strd[nd-1]
	t2Stride := t2.strd[nd-1]

	dims := util.DotDims(t1.dims)
	o := &CPUTensor{
		dims: dims,
		strd: util.DimsToStrides(dims),
		data: make([]float64, util.DimsToNumElems(dims)),
	}

	kernel := func(t1Off, t2Off, oOff int) {
		for k := range n {
			o.data[oOff] += t1.data[t1Off+k*t1Stride] * t2.data[t2Off+k*t2Stride]
		}
	}

	nb := nd - 1
	if nb == 0 {
		kernel(0, 0, 0)
		return o
	}

	batchIdx := make([]int, nb)
	numBatch := util.DimsToNumElems(dims)
	for range numBatch {
		t1Off, t2Off, oOff := 0, 0, 0
		for d := range nb {
			t1Off += batchIdx[d] * t1.strd[d]
			t2Off += batchIdx[d] * t2.strd[d]
			oOff += batchIdx[d] * o.strd[d]
		}
		kernel(t1Off, t2Off, oOff)
		updateElementWiseIndex(batchIdx, dims)
	}

	return o
}

func (t *CPUTensor) matMul(u *CPUTensor) *CPUTensor {
	t1, t2 := t, u
	nd := len(t1.dims)

	m := t1.dims[nd-2]
	n := t1.dims[nd-1]
	k := t2.dims[nd-1]

	t1RowStride := t1.strd[nd-2]
	t1ColStride := t1.strd[nd-1]
	t2RowStride := t2.strd[nd-2]
	t2ColStride := t2.strd[nd-1]

	dims := util.MatMulDims(t1.dims, t2.dims)
	o := &CPUTensor{
		dims: dims,
		strd: util.DimsToStrides(dims),
		data: make([]float64, util.DimsToNumElems(dims)),
	}
	oRowStride := o.strd[nd-2]

	kernel := func(t1Off, t2Off, oOff int) {
		for i := range m {
			t1Row := t1Off + i*t1RowStride
			oRow := oOff + i*oRowStride
			for p := range n {
				a := t1.data[t1Row+p*t1ColStride]
				t2Row := t2Off + p*t2RowStride
				for j := range k {
					o.data[oRow+j] += a * t2.data[t2Row+j*t2ColStride]
				}
			}
		}
	}

	nb := nd - 2
	if nb == 0 {
		kernel(0, 0, 0)
		return o
	}

	batchDims := dims[:nb]
	batchIdx := make([]int, nb)
	numBatch := util.DimsToNumElems(batchDims)
	for range numBatch {
		t1Off, t2Off, oOff := 0, 0, 0
		for d := range nb {
			t1Off += batchIdx[d] * t1.strd[d]
			t2Off += batchIdx[d] * t2.strd[d]
			oOff += batchIdx[d] * o.strd[d]
		}
		kernel(t1Off, t2Off, oOff)
		updateElementWiseIndex(batchIdx, batchDims)
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
