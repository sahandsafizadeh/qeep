package cputensor

import "math"

const float64EqualityThreshold = 1e-240

func (t *CPUTensor) scale(u float64) (o *CPUTensor) {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return u * a })
}

func (t *CPUTensor) pow(u float64) (o *CPUTensor) {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Pow(a, u) })
}

func (t *CPUTensor) exp() (o *CPUTensor) {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Exp(a) })
}

func (t *CPUTensor) log() (o *CPUTensor) {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Log(a) })
}

func (t *CPUTensor) sin() (o *CPUTensor) {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Sin(a) })
}

func (t *CPUTensor) cos() (o *CPUTensor) {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Cos(a) })
}

func (t *CPUTensor) tan() (o *CPUTensor) {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Tan(a) })
}

func (t *CPUTensor) sinh() (o *CPUTensor) {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Sinh(a) })
}

func (t *CPUTensor) cosh() (o *CPUTensor) {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Cosh(a) })
}

func (t *CPUTensor) tanh() (o *CPUTensor) {
	return applyUnaryFuncOnTensorElemWise(t, func(a float64) float64 { return math.Tanh(a) })
}

func (t *CPUTensor) eq(u *CPUTensor) (o *CPUTensor) {
	return applyBinaryFuncOnTensorsElemWise(t, u,
		func(a, b float64) float64 {
			if math.Abs(a-b) <= float64EqualityThreshold {
				return 1.
			} else {
				return 0.
			}
		})
}

func (t *CPUTensor) ne(u *CPUTensor) (o *CPUTensor) {
	return applyBinaryFuncOnTensorsElemWise(t, u,
		func(a, b float64) float64 {
			if math.Abs(a-b) <= float64EqualityThreshold {
				return 0.
			} else {
				return 1.
			}
		})
}

func (t *CPUTensor) gt(u *CPUTensor) (o *CPUTensor) {
	return applyBinaryFuncOnTensorsElemWise(t, u,
		func(a, b float64) float64 {
			if a > b {
				return 1.
			} else {
				return 0.
			}
		})
}

func (t *CPUTensor) ge(u *CPUTensor) (o *CPUTensor) {
	return applyBinaryFuncOnTensorsElemWise(t, u,
		func(a, b float64) float64 {
			if a >= b {
				return 1.
			} else {
				return 0.
			}
		})
}

func (t *CPUTensor) lt(u *CPUTensor) (o *CPUTensor) {
	return applyBinaryFuncOnTensorsElemWise(t, u,
		func(a, b float64) float64 {
			if a < b {
				return 1.
			} else {
				return 0.
			}
		})
}

func (t *CPUTensor) le(u *CPUTensor) (o *CPUTensor) {
	return applyBinaryFuncOnTensorsElemWise(t, u,
		func(a, b float64) float64 {
			if a <= b {
				return 1.
			} else {
				return 0.
			}
		})
}

func (t *CPUTensor) elmax(u *CPUTensor) (o *CPUTensor) {
	return applyBinaryFuncOnTensorsElemWise(t, u, func(a, b float64) float64 { return math.Max(a, b) })
}

func (t *CPUTensor) elmin(u *CPUTensor) (o *CPUTensor) {
	return applyBinaryFuncOnTensorsElemWise(t, u, func(a, b float64) float64 { return math.Min(a, b) })
}

func (t *CPUTensor) add(u *CPUTensor) (o *CPUTensor) {
	return applyBinaryFuncOnTensorsElemWise(t, u, func(a, b float64) float64 { return a + b })
}

func (t *CPUTensor) sub(u *CPUTensor) (o *CPUTensor) {
	return applyBinaryFuncOnTensorsElemWise(t, u, func(a, b float64) float64 { return a - b })
}

func (t *CPUTensor) mul(u *CPUTensor) (o *CPUTensor) {
	return applyBinaryFuncOnTensorsElemWise(t, u, func(a, b float64) float64 { return a * b })
}

func (t *CPUTensor) div(u *CPUTensor) (o *CPUTensor) {
	return applyBinaryFuncOnTensorsElemWise(t, u, func(a, b float64) float64 { return a / b })
}

func (t *CPUTensor) dot(u *CPUTensor) (o *CPUTensor) {
	t1, t2 := t, u
	dims := dotDims(t1.dims)
	elemGen := linearLastDimDotProductElemGenerator(t1, t2)

	o = new(CPUTensor)
	o.dims = dims
	o.initWith(elemGen)

	return o
}

func (t *CPUTensor) matMul(u *CPUTensor) (o *CPUTensor) {
	t1, t2 := t, u
	td := len(t1.dims)
	dims := matMulDims(t1.dims, t2.dims)
	elemGen := linearLast2DimsMatMulElemGenerator(t1, t2)

	o = new(CPUTensor)
	o.dims = dims[:td-2]
	o.initWith(elemGen)

	o.dims = dims
	return o
}

func (t *CPUTensor) equals(u *CPUTensor) (are bool) {
	o := t.eq(u)
	n := o.numElems()
	return o.sum() >= float64(n)
}

func applyUnaryFuncOnTensorElemWise(t *CPUTensor, suf scalarUnaryFunc) (o *CPUTensor) {

	var calcData func([]int, *any, *any)
	calcData = func(dims []int, a, r *any) {
		if len(dims) == 0 {
			*r = suf((*a).(float64))
			return
		}

		aRows := (*a).([]any)
		rRows := make([]any, dims[0])
		dims = dims[1:]

		for i := range rRows {
			calcData(dims, &aRows[i], &rRows[i])
		}

		*r = rRows
	}

	o = new(CPUTensor)
	o.dims = make([]int, len(t.dims))
	copy(o.dims, t.dims)
	calcData(t.dims, &t.data, &o.data)

	return o
}

func applyBinaryFuncOnTensorsElemWise(t1, t2 *CPUTensor, sbf scalarBinaryFunc) (o *CPUTensor) {

	var calcData func([]int, *any, *any, *any)
	calcData = func(dims []int, a, b, r *any) {
		if len(dims) == 0 {
			*r = sbf((*a).(float64), (*b).(float64))
			return
		}

		aRows := (*a).([]any)
		bRows := (*b).([]any)
		rRows := make([]any, dims[0])
		dims = dims[1:]

		for i := range rRows {
			calcData(dims, &aRows[i], &bRows[i], &rRows[i])
		}

		*r = rRows
	}

	o = new(CPUTensor)
	o.dims = make([]int, len(t1.dims))
	copy(o.dims, t1.dims)
	calcData(t1.dims, &t1.data, &t2.data, &o.data)

	return o
}

/* ----- helpers ----- */

func linearLastDimDotProductElemGenerator(t1, t2 *CPUTensor) initializerFunc {
	dims := t1.dims
	n := len(dims) - 1
	state := make([]int, n)

	return func() any {
		data1 := t1.dataAt(state)
		data2 := t2.dataAt(state)
		prodRes := dotProductOf1DInputs(data1, data2)

		i := n - 1
		for i >= 0 {
			if state[i] < dims[i]-1 {
				state[i]++
				break
			} else {
				state[i] = 0
				i--
			}
		}

		return prodRes
	}
}

func linearLast2DimsMatMulElemGenerator(t1, t2 *CPUTensor) initializerFunc {
	dims := t1.dims
	n := len(dims) - 2
	state := make([]int, n)

	return func() any {
		data1 := t1.dataAt(state)
		data2 := t2.dataAt(state)
		mulRes := matMulDataOf2DInputs(data1, data2)

		i := n - 1
		for i >= 0 {
			if state[i] < dims[i]-1 {
				state[i]++
				break
			} else {
				state[i] = 0
				i--
			}
		}

		return mulRes
	}
}

func dotProductOf1DInputs(a, b any) (c any) {
	v1 := a.([]any)
	v2 := b.([]any)
	n := len(v1)

	s := 0.
	var i int
	for i = range n {
		eiv1 := v1[i].(float64)
		eiv2 := v2[i].(float64)
		s += eiv1 * eiv2
	}

	return s
}

func matMulDataOf2DInputs(a, b any) (c any) {
	m1 := a.([]any)
	m2 := b.([]any)
	r0m1 := m1[0].([]any)
	r0m2 := m2[0].([]any)

	// A_mn * B_nk = C_mk
	m := len(m1)
	n := len(r0m1)
	k := len(r0m2)

	var i, j, p int
	cRows := make([]any, m)
	for i = range m {
		row := make([]any, k)
		for j = range k {
			eij := 0.
			for p = range n {
				rim1 := m1[i].([]any)
				rpm2 := m2[p].([]any)
				eipm1 := rim1[p].(float64)
				epjm2 := rpm2[j].(float64)
				eij += eipm1 * epjm2
			}
			row[j] = eij
		}
		cRows[i] = row
	}

	return cRows
}

func dotDims(idims []int) (dims []int) {
	td := len(idims)
	cd := idims[:td-1]
	dims = make([]int, len(cd))
	copy(dims, cd)

	return dims
}

func matMulDims(dims1, dims2 []int) (dims []int) {
	td := len(dims1)
	cd := dims1[:td-2]
	dims = make([]int, len(cd))
	copy(dims, cd)

	m := dims1[td-2]
	k := dims2[td-1]
	dims = append(dims, m, k)

	return dims
}
