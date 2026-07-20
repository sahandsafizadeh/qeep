//go:build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR}/lib -Wl,-rpath=${SRCDIR}/lib -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import "github.com/sahandsafizadeh/qeep/tensor/internal/util"

func (t *CUDATensor) scale(u float64) *CUDATensor {
	return applyHalfBinaryOperation(t, u, t.dims, func(x C.CUDATensor, a C.double, view_o C.CUDAView) *C.double {
		return C.Scale(x, a, view_o)
	})
}

func (t *CUDATensor) pow(u float64) *CUDATensor {
	return applyHalfBinaryOperation(t, u, t.dims, func(x C.CUDATensor, a C.double, view_o C.CUDAView) *C.double {
		return C.Pow(x, a, view_o)
	})
}

func (t *CUDATensor) exp() *CUDATensor {
	return applyUnaryOperation(t, t.dims, func(x C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Exp(x, view_o)
	})
}

func (t *CUDATensor) log() *CUDATensor {
	return applyUnaryOperation(t, t.dims, func(x C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Log(x, view_o)
	})
}

func (t *CUDATensor) sin() *CUDATensor {
	return applyUnaryOperation(t, t.dims, func(x C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Sin(x, view_o)
	})
}

func (t *CUDATensor) cos() *CUDATensor {
	return applyUnaryOperation(t, t.dims, func(x C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Cos(x, view_o)
	})
}

func (t *CUDATensor) tan() *CUDATensor {
	return applyUnaryOperation(t, t.dims, func(x C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Tan(x, view_o)
	})
}

func (t *CUDATensor) sinh() *CUDATensor {
	return applyUnaryOperation(t, t.dims, func(x C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Sinh(x, view_o)
	})
}

func (t *CUDATensor) cosh() *CUDATensor {
	return applyUnaryOperation(t, t.dims, func(x C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Cosh(x, view_o)
	})
}

func (t *CUDATensor) tanh() *CUDATensor {
	return applyUnaryOperation(t, t.dims, func(x C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Tanh(x, view_o)
	})
}

func (t *CUDATensor) eq(u *CUDATensor) *CUDATensor {
	return applyBinaryOperation(t, u, t.dims, func(a C.CUDATensor, b C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Eq(a, b, view_o)
	})
}

func (t *CUDATensor) ne(u *CUDATensor) *CUDATensor {
	return applyBinaryOperation(t, u, t.dims, func(a C.CUDATensor, b C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Ne(a, b, view_o)
	})
}

func (t *CUDATensor) gt(u *CUDATensor) *CUDATensor {
	return applyBinaryOperation(t, u, t.dims, func(a C.CUDATensor, b C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Gt(a, b, view_o)
	})
}

func (t *CUDATensor) ge(u *CUDATensor) *CUDATensor {
	return applyBinaryOperation(t, u, t.dims, func(a C.CUDATensor, b C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Ge(a, b, view_o)
	})
}

func (t *CUDATensor) lt(u *CUDATensor) *CUDATensor {
	return applyBinaryOperation(t, u, t.dims, func(a C.CUDATensor, b C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Lt(a, b, view_o)
	})
}

func (t *CUDATensor) le(u *CUDATensor) *CUDATensor {
	return applyBinaryOperation(t, u, t.dims, func(a C.CUDATensor, b C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Le(a, b, view_o)
	})
}

func (t *CUDATensor) elmax(u *CUDATensor) *CUDATensor {
	return applyBinaryOperation(t, u, t.dims, func(a C.CUDATensor, b C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.ElMax(a, b, view_o)
	})
}

func (t *CUDATensor) elmin(u *CUDATensor) *CUDATensor {
	return applyBinaryOperation(t, u, t.dims, func(a C.CUDATensor, b C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.ElMin(a, b, view_o)
	})
}

func (t *CUDATensor) add(u *CUDATensor) *CUDATensor {
	return applyBinaryOperation(t, u, t.dims, func(a C.CUDATensor, b C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Add(a, b, view_o)
	})
}

func (t *CUDATensor) sub(u *CUDATensor) *CUDATensor {
	return applyBinaryOperation(t, u, t.dims, func(a C.CUDATensor, b C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Sub(a, b, view_o)
	})
}

func (t *CUDATensor) mul(u *CUDATensor) *CUDATensor {
	return applyBinaryOperation(t, u, t.dims, func(a C.CUDATensor, b C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Mul(a, b, view_o)
	})
}

func (t *CUDATensor) div(u *CUDATensor) *CUDATensor {
	return applyBinaryOperation(t, u, t.dims, func(a C.CUDATensor, b C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Div(a, b, view_o)
	})
}

func (t *CUDATensor) dot(u *CUDATensor) *CUDATensor {
	dims := util.DotDims(t.dims)

	return applyBinaryOperation(t, u, dims, func(a C.CUDATensor, b C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.Dot(a, b, view_o)
	})
}

func (t *CUDATensor) matMul(u *CUDATensor) *CUDATensor {
	dims := util.MatMulDims(t.dims, u.dims)

	return applyBinaryOperation(t, u, dims, func(a C.CUDATensor, b C.CUDATensor, view_o C.CUDAView) *C.double {
		return C.MatMul(a, b, view_o)
	})
}

func (t *CUDATensor) equals(u *CUDATensor) bool {
	o := t.eq(u)
	n := o.numElems()
	return o.sum() >= float64(n)
}

func applyHalfBinaryOperation(x *CUDATensor, a float64, dims []int, hbf_c halfBinaryOperatorFunc_C) *CUDATensor {
	x_c := toCUDATensor_C(x)
	a_c := (C.double)(a)
	view_o_c := toCUDAView_C(dims)

	data_c := hbf_c(x_c, a_c, view_o_c)

	return newCUDATensor(dims, data_c)
}

func applyUnaryOperation(x *CUDATensor, dims []int, uf_c unaryOperatorFunc_C) *CUDATensor {
	x_c := toCUDATensor_C(x)
	view_o_c := toCUDAView_C(dims)

	data_c := uf_c(x_c, view_o_c)

	return newCUDATensor(dims, data_c)
}

func applyBinaryOperation(a *CUDATensor, b *CUDATensor, dims []int, bf_c binaryOperatorFunc_C) *CUDATensor {
	a_c := toCUDATensor_C(a)
	b_c := toCUDATensor_C(b)
	view_o_c := toCUDAView_C(dims)

	data_c := bf_c(a_c, b_c, view_o_c)

	return newCUDATensor(dims, data_c)
}
