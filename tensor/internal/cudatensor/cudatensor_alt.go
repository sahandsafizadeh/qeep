//go:build !cuda
// +build !cuda

package cudatensor

import "github.com/sahandsafizadeh/qeep/tensor/internal/tensor"

const message = "unavailable implementation: tensors on CUDA device require 'cuda' build tag"

func Full(dims []int, value float64, withGrad bool) (o tensor.Tensor, err error) {
	panic(message)
}

func Zeros(dims []int, withGrad bool) (o tensor.Tensor, err error) {
	panic(message)
}

func Ones(dims []int, withGrad bool) (o tensor.Tensor, err error) {
	panic(message)
}

func Eye(d int, withGrad bool) (o tensor.Tensor, err error) {
	panic(message)
}

func RandU(dims []int, l, u float64, withGrad bool) (o tensor.Tensor, err error) {
	panic(message)
}

func RandN(dims []int, u, s float64, withGrad bool) (o tensor.Tensor, err error) {
	panic(message)
}

func Of(data any, withGrad bool) (o tensor.Tensor, err error) {
	panic(message)
}

func Concat(ts []tensor.Tensor, dim int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) NElems() (n int) {
	panic(message)
}

func (t *CUDATensor) Shape() (shape []int) {
	panic(message)
}

func (t *CUDATensor) At(index ...int) (value float64, err error) {
	panic(message)
}

func (t *CUDATensor) Slice(index []tensor.Range) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Patch(index []tensor.Range, u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Transpose() (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Reshape(shape []int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) UnSqueeze(dim int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Squeeze(dim int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Flatten(fromDim int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Broadcast(shape []int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Sum() (value float64) {
	panic(message)
}

func (t *CUDATensor) Max() (value float64) {
	panic(message)
}

func (t *CUDATensor) Min() (value float64) {
	panic(message)
}

func (t *CUDATensor) Avg() (value float64) {
	panic(message)
}

func (t *CUDATensor) Var() (value float64) {
	panic(message)
}

func (t *CUDATensor) Std() (value float64) {
	panic(message)
}

func (t *CUDATensor) Mean() (value float64) {
	panic(message)
}

func (t *CUDATensor) Argmax(dim int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Argmin(dim int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) SumAlong(dim int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) MaxAlong(dim int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) MinAlong(dim int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) AvgAlong(dim int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) VarAlong(dim int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) StdAlong(dim int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) MeanAlong(dim int) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Scale(u float64) (o tensor.Tensor) {
	panic(message)
}

func (t *CUDATensor) Pow(u float64) (o tensor.Tensor) {
	panic(message)
}

func (t *CUDATensor) Exp() (o tensor.Tensor) {
	panic(message)
}

func (t *CUDATensor) Log() (o tensor.Tensor) {
	panic(message)
}

func (t *CUDATensor) Sin() (o tensor.Tensor) {
	panic(message)
}

func (t *CUDATensor) Cos() (o tensor.Tensor) {
	panic(message)
}

func (t *CUDATensor) Tan() (o tensor.Tensor) {
	panic(message)
}

func (t *CUDATensor) Sinh() (o tensor.Tensor) {
	panic(message)
}

func (t *CUDATensor) Cosh() (o tensor.Tensor) {
	panic(message)
}

func (t *CUDATensor) Tanh() (o tensor.Tensor) {
	panic(message)
}

func (t *CUDATensor) Eq(u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Ne(u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Gt(u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Ge(u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Lt(u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Le(u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) ElMax(u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) ElMin(u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Add(u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Sub(u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Mul(u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Div(u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Dot(u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) MatMul(u tensor.Tensor) (o tensor.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Equals(u tensor.Tensor) (are bool, err error) {
	panic(message)
}

func (t *CUDATensor) GradContext() (gctx any) {
	panic(message)
}

func (t *CUDATensor) ResetGradContext(tracked bool) {
	panic(message)
}

func (t *CUDATensor) Gradient() (g tensor.Tensor) {
	panic(message)
}
