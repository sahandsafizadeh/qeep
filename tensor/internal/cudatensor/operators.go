package cudatensor

/*
   #cgo LDFLAGS: -L${SRCDIR} -lcudatensor
   #include "cuda_c/cudatensor.h"
*/
import "C"

func (t *CUDATensor) scale(u float64) (o *CUDATensor) {
	return applyHalfBinaryOperation(t, u, func(x *C.double, n C.size_t, a C.double) (y *C.double) {
		return C.Scale(x, n, a)
	})
}

func (t *CUDATensor) pow(u float64) (o *CUDATensor) {
	return applyHalfBinaryOperation(t, u, func(x *C.double, n C.size_t, a C.double) (y *C.double) {
		return C.Pow(x, n, a)
	})
}

func (t *CUDATensor) exp() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x *C.double, n C.size_t) (y *C.double) {
		return C.Exp(x, n)
	})
}

func (t *CUDATensor) log() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x *C.double, n C.size_t) (y *C.double) {
		return C.Log(x, n)
	})
}

func (t *CUDATensor) sin() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x *C.double, n C.size_t) (y *C.double) {
		return C.Sin(x, n)
	})
}

func (t *CUDATensor) cos() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x *C.double, n C.size_t) (y *C.double) {
		return C.Cos(x, n)
	})
}

func (t *CUDATensor) tan() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x *C.double, n C.size_t) (y *C.double) {
		return C.Tan(x, n)
	})
}

func (t *CUDATensor) sinh() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x *C.double, n C.size_t) (y *C.double) {
		return C.Sinh(x, n)
	})
}

func (t *CUDATensor) cosh() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x *C.double, n C.size_t) (y *C.double) {
		return C.Cosh(x, n)
	})
}

func (t *CUDATensor) tanh() (o *CUDATensor) {
	return applyUnaryOperation(t, func(x *C.double, n C.size_t) (y *C.double) {
		return C.Tanh(x, n)
	})
}

func (t *CUDATensor) eq(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a *C.double, b *C.double, n C.size_t) (c *C.double) {
		return C.Eq(a, b, n)
	})
}

func (t *CUDATensor) ne(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a *C.double, b *C.double, n C.size_t) (c *C.double) {
		return C.Ne(a, b, n)
	})
}

func (t *CUDATensor) gt(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a *C.double, b *C.double, n C.size_t) (c *C.double) {
		return C.Gt(a, b, n)
	})
}

func (t *CUDATensor) ge(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a *C.double, b *C.double, n C.size_t) (c *C.double) {
		return C.Ge(a, b, n)
	})
}

func (t *CUDATensor) lt(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a *C.double, b *C.double, n C.size_t) (c *C.double) {
		return C.Lt(a, b, n)
	})
}

func (t *CUDATensor) le(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a *C.double, b *C.double, n C.size_t) (c *C.double) {
		return C.Le(a, b, n)
	})
}

func (t *CUDATensor) elmax(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a *C.double, b *C.double, n C.size_t) (c *C.double) {
		return C.ElMax(a, b, n)
	})
}

func (t *CUDATensor) elmin(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a *C.double, b *C.double, n C.size_t) (c *C.double) {
		return C.ElMin(a, b, n)
	})
}

func (t *CUDATensor) add(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a *C.double, b *C.double, n C.size_t) (c *C.double) {
		return C.Add(a, b, n)
	})
}

func (t *CUDATensor) sub(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a *C.double, b *C.double, n C.size_t) (c *C.double) {
		return C.Sub(a, b, n)
	})
}

func (t *CUDATensor) mul(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a *C.double, b *C.double, n C.size_t) (c *C.double) {
		return C.Mul(a, b, n)
	})
}

func (t *CUDATensor) div(u *CUDATensor) (o *CUDATensor) {
	return applyBinaryOperation(t, u, func(a *C.double, b *C.double, n C.size_t) (c *C.double) {
		return C.Div(a, b, n)
	})
}

func (t *CUDATensor) equals(u *CUDATensor) (are bool) {
	o := t.eq(u)
	n := o.n
	return o.sum() >= float64(n)
}

func applyUnaryOperation(x *CUDATensor, cuf cudacUnaryFunc) (y *CUDATensor) {
	dims := x.dims
	data := x.data
	n := dimsToNumElems(dims)

	x_c := (*C.double)(data)
	n_c := (C.size_t)(n)

	data_c := cuf(x_c, n_c)

	return newCUDATensor(dims, data_c)
}

func applyBinaryOperation(a *CUDATensor, b *CUDATensor, cbf cudacBinaryFunc) (c *CUDATensor) {
	dims := a.dims
	data1 := a.data
	data2 := b.data
	n := dimsToNumElems(dims)

	a_c := (*C.double)(data1)
	b_c := (*C.double)(data2)
	n_c := (C.size_t)(n)

	data_c := cbf(a_c, b_c, n_c)

	return newCUDATensor(dims, data_c)
}

func applyHalfBinaryOperation(x *CUDATensor, a float64, chbf cudacHalfBinaryFunc) (y *CUDATensor) {
	dims := x.dims
	data := x.data
	n := dimsToNumElems(dims)

	x_c := (*C.double)(data)
	n_c := (C.size_t)(n)
	a_c := (C.double)(a)

	data_c := chbf(x_c, n_c, a_c)

	return newCUDATensor(dims, data_c)
}
