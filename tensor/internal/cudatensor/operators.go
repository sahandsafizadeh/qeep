//go:build cuda
// +build cuda

package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR}/lib -Wl,-rpath=${SRCDIR}/lib -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

import "github.com/sahandsafizadeh/qeep/tensor/internal/util"

func (t *CUDATensor) scale(u float64) (o *CUDATensor) {
	return applyHalfBinaryOperation(t, u, func(x C.CudaData, a C.double) (y *C.double) {
		return C.Scale(x, a)
	})
}

func (t *CUDATensor) pow(u float64) (o *CUDATensor) {
	return applyHalfBinaryOperation(t, u, func(x C.CudaData, a C.double) (y *C.double) {
		return C.Pow(x, a)
	})
}

func (t *CUDATensor) exp() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x C.CudaData) (y *C.double) {
		return C.Exp(x)
	})
}

func (t *CUDATensor) log() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x C.CudaData) (y *C.double) {
		return C.Log(x)
	})
}

func (t *CUDATensor) sin() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x C.CudaData) (y *C.double) {
		return C.Sin(x)
	})
}

func (t *CUDATensor) cos() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x C.CudaData) (y *C.double) {
		return C.Cos(x)
	})
}

func (t *CUDATensor) tan() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x C.CudaData) (y *C.double) {
		return C.Tan(x)
	})
}

func (t *CUDATensor) sinh() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x C.CudaData) (y *C.double) {
		return C.Sinh(x)
	})
}

func (t *CUDATensor) cosh() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x C.CudaData) (y *C.double) {
		return C.Cosh(x)
	})
}

func (t *CUDATensor) tanh() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x C.CudaData) (y *C.double) {
		return C.Tanh(x)
	})
}

func (t *CUDATensor) eq(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a C.CudaData, b C.CudaData) (c *C.double) {
		return C.Eq(a, b)
	})
}

func (t *CUDATensor) ne(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a C.CudaData, b C.CudaData) (c *C.double) {
		return C.Ne(a, b)
	})
}

func (t *CUDATensor) gt(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a C.CudaData, b C.CudaData) (c *C.double) {
		return C.Gt(a, b)
	})
}

func (t *CUDATensor) ge(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a C.CudaData, b C.CudaData) (c *C.double) {
		return C.Ge(a, b)
	})
}

func (t *CUDATensor) lt(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a C.CudaData, b C.CudaData) (c *C.double) {
		return C.Lt(a, b)
	})
}

func (t *CUDATensor) le(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a C.CudaData, b C.CudaData) (c *C.double) {
		return C.Le(a, b)
	})
}

func (t *CUDATensor) elmax(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a C.CudaData, b C.CudaData) (c *C.double) {
		return C.ElMax(a, b)
	})
}

func (t *CUDATensor) elmin(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a C.CudaData, b C.CudaData) (c *C.double) {
		return C.ElMin(a, b)
	})
}

func (t *CUDATensor) add(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a C.CudaData, b C.CudaData) (c *C.double) {
		return C.Add(a, b)
	})
}

func (t *CUDATensor) sub(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a C.CudaData, b C.CudaData) (c *C.double) {
		return C.Sub(a, b)
	})
}

func (t *CUDATensor) mul(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a C.CudaData, b C.CudaData) (c *C.double) {
		return C.Mul(a, b)
	})
}

func (t *CUDATensor) div(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a C.CudaData, b C.CudaData) (c *C.double) {
		return C.Div(a, b)
	})
}

func (t *CUDATensor) dot(u *CUDATensor) (o *CUDATensor) {
	return applyDot(t, u)
}

func (t *CUDATensor) matMul(u *CUDATensor) (o *CUDATensor) {
	return applyMatMul(t, u)
}

func (t *CUDATensor) equals(u *CUDATensor) (are bool) {
	o := t.eq(u)
	n := o.n
	return o.sum() >= float64(n)
}

func applyHalfBinaryOperation(x *CUDATensor, a float64, chbf cudacHalfBinaryFunc) (y *CUDATensor) {
	x_c := getCudaDataOf(x)
	a_c := (C.double)(a)

	data_c := chbf(x_c, a_c)

	return newCUDATensor(x.dims, data_c)
}

func applyUnaryOperation(x *CUDATensor, cuf cudacUnaryFunc) (y *CUDATensor) {
	x_c := getCudaDataOf(x)

	data_c := cuf(x_c)

	return newCUDATensor(x.dims, data_c)
}

func applyBinaryOperation(a *CUDATensor, b *CUDATensor, cbf cudacBinaryFunc) (c *CUDATensor) {
	a_c := getCudaDataOf(a)
	b_c := getCudaDataOf(b)

	data_c := cbf(a_c, b_c)

	return newCUDATensor(a.dims, data_c)
}

func applyDot(a *CUDATensor, b *CUDATensor) (c *CUDATensor) {
	dims := util.DotDims(a.dims)

	a_c := getCudaDataOf(a)
	b_c := getCudaDataOf(b)
	dims_src_c := getDimArrOf(a.dims)
	dims_dst_c := getDimArrOf(dims)

	data_c := C.Dot(a_c, b_c, dims_src_c, dims_dst_c)

	return newCUDATensor(dims, data_c)
}

func applyMatMul(a *CUDATensor, b *CUDATensor) (c *CUDATensor) {
	dims := util.MatMulDims(a.dims, b.dims)

	a_c := getCudaDataOf(a)
	b_c := getCudaDataOf(b)
	dims_a_c := getDimArrOf(a.dims)
	dims_b_c := getDimArrOf(b.dims)
	dims_c_c := getDimArrOf(dims)

	data_c := C.MatMul(a_c, b_c, dims_a_c, dims_b_c, dims_c_c)

	return newCUDATensor(dims, data_c)
}
