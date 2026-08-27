//go:build !cuda

package cudatensor

import "github.com/sahandsafizadeh/qeep/tensor/internal/core"

// used for device selection at tests
const IsAvailable = false

const message = `CUDA implementation for tensors is not available:
(1) make sure you have a working device
(2) make sure you have the CUDA toolkit installed
(3) build CUDA libraries by running "make cuda"
(4) use 'cuda' build tag in the go tool
`

func Full(dims []int, value float64, withGrad bool) (o core.Tensor, err error) {
	panic(message)
}

func Zeros(dims []int, withGrad bool) (o core.Tensor, err error) {
	panic(message)
}

func Ones(dims []int, withGrad bool) (o core.Tensor, err error) {
	panic(message)
}

func Eye(d int, withGrad bool) (o core.Tensor, err error) {
	panic(message)
}

func RandU(dims []int, l, u float64, withGrad bool) (o core.Tensor, err error) {
	panic(message)
}

func RandN(dims []int, u, s float64, withGrad bool) (o core.Tensor, err error) {
	panic(message)
}

func Of(data any, withGrad bool) (o core.Tensor, err error) {
	panic(message)
}

func Concat(ts []core.Tensor, dim int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) NElems() int {
	panic(message)
}

func (t *CUDATensor) Shape() []int {
	panic(message)
}

func (t *CUDATensor) At(index ...int) (value float64, err error) {
	panic(message)
}

func (t *CUDATensor) Slice(index []core.Range) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Patch(index []core.Range, u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Transpose() (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Reshape(shape []int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) UnSqueeze(dim int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Squeeze(dim int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Flatten(fromDim int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Broadcast(shape []int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Sum() float64 {
	panic(message)
}

func (t *CUDATensor) Max() float64 {
	panic(message)
}

func (t *CUDATensor) Min() float64 {
	panic(message)
}

func (t *CUDATensor) Avg() float64 {
	panic(message)
}

func (t *CUDATensor) Var() float64 {
	panic(message)
}

func (t *CUDATensor) Std() float64 {
	panic(message)
}

func (t *CUDATensor) Mean() float64 {
	panic(message)
}

func (t *CUDATensor) Argmax(dim int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Argmin(dim int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) SumAlong(dim int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) MaxAlong(dim int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) MinAlong(dim int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) AvgAlong(dim int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) VarAlong(dim int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) StdAlong(dim int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) MeanAlong(dim int) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Scale(u float64) core.Tensor {
	panic(message)
}

func (t *CUDATensor) Pow(u float64) core.Tensor {
	panic(message)
}

func (t *CUDATensor) Exp() core.Tensor {
	panic(message)
}

func (t *CUDATensor) Log() core.Tensor {
	panic(message)
}

func (t *CUDATensor) Sin() core.Tensor {
	panic(message)
}

func (t *CUDATensor) Cos() core.Tensor {
	panic(message)
}

func (t *CUDATensor) Tan() core.Tensor {
	panic(message)
}

func (t *CUDATensor) Sinh() core.Tensor {
	panic(message)
}

func (t *CUDATensor) Cosh() core.Tensor {
	panic(message)
}

func (t *CUDATensor) Tanh() core.Tensor {
	panic(message)
}

func (t *CUDATensor) Eq(u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Ne(u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Gt(u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Ge(u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Lt(u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Le(u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) ElMax(u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) ElMin(u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Add(u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Sub(u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Mul(u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Div(u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Dot(u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) MatMul(u core.Tensor) (o core.Tensor, err error) {
	panic(message)
}

func (t *CUDATensor) Equals(u core.Tensor) (are bool, err error) {
	panic(message)
}

func (t *CUDATensor) Gradient() core.Tensor {
	panic(message)
}

func (t *CUDATensor) GradientTracked() bool {
	panic(message)
}

func (t *CUDATensor) ResetGradContext(tracked bool) {
	panic(message)
}

func (t *CUDATensor) GradContext() any {
	panic(message)
}

func (t *CUDATensor) Device() core.Device {
	panic(message)
}
