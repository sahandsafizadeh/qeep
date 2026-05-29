package cputensor

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
	"github.com/sahandsafizadeh/qeep/tensor/internal/validator"
)

/*------------- initializers ------------*/

func Full(dims []int, value float64, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims(dims)
	if err != nil {
		return o, fmt.Errorf("Full input dimension validation failed: %w", err)
	}

	r := constTensor(dims, value)
	r.gctx = gradtrack.NewGradContext(withGrad)

	return r, nil
}

func Zeros(dims []int, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims(dims)
	if err != nil {
		return o, fmt.Errorf("Zeros input dimension validation failed: %w", err)
	}

	r := constTensor(dims, 0.)
	r.gctx = gradtrack.NewGradContext(withGrad)

	return r, nil
}

func Ones(dims []int, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims(dims)
	if err != nil {
		return o, fmt.Errorf("Ones input dimension validation failed: %w", err)
	}

	r := constTensor(dims, 1.)
	r.gctx = gradtrack.NewGradContext(withGrad)

	return r, nil
}

func Eye(d int, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims([]int{d, d})
	if err != nil {
		return o, fmt.Errorf("Eye input dimension validation failed: %w", err)
	}

	r := eyeMatrix(d)
	r.gctx = gradtrack.NewGradContext(withGrad)

	return r, nil
}

func RandU(dims []int, l, u float64, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateRandUParams(l, u)
	if err != nil {
		return o, fmt.Errorf("RandU random parameter validation failed: %w", err)
	}

	err = validator.ValidateInputDims(dims)
	if err != nil {
		return o, fmt.Errorf("RandU input dimension validation failed: %w", err)
	}

	r := uniformRandomTensor(dims, l, u)
	r.gctx = gradtrack.NewGradContext(withGrad)

	return r, nil
}

func RandN(dims []int, u, s float64, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateRandNParams(u, s)
	if err != nil {
		return o, fmt.Errorf("RandN random parameter validation failed: %w", err)
	}

	err = validator.ValidateInputDims(dims)
	if err != nil {
		return o, fmt.Errorf("RandN input dimension validation failed: %w", err)
	}

	r := normalRandomTensor(dims, u, s)
	r.gctx = gradtrack.NewGradContext(withGrad)

	return r, nil
}

func Of(data any, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDataDimUnity(data)
	if err != nil {
		return o, fmt.Errorf("Of input data validation failed: %w", err)
	}

	r := tensorFromData(data)
	r.gctx = gradtrack.NewGradContext(withGrad)

	return r, nil
}

func Concat(ts []tensor.Tensor, dim int) (o tensor.Tensor, err error) {
	_ts, err := assertCPUTensors(ts)
	if err != nil {
		return o, fmt.Errorf("Concat tensors' device validation failed: %w", err)
	}

	tsDims := make([][]int, len(_ts))
	for i, cu := range _ts {
		tsDims[i] = cu.dims
	}

	err = validator.ValidateConcatTensorsDimsAlongDim(tsDims, dim)
	if err != nil {
		return o, fmt.Errorf("Concat inputs' dimension validation failed: %w", err)
	}

	r := tensorFromConcat(_ts, dim)
	r.gctx = gradtrack.Concat(r, ts, dim)

	return r, nil
}

/*--------------- methods ---------------*/

func (t *CPUTensor) NElems() int {
	return t.numElems()
}

func (t *CPUTensor) Shape() []int {
	shape := make([]int, len(t.dims))
	copy(shape, t.dims)
	return shape
}

func (t *CPUTensor) At(index ...int) (value float64, err error) {
	err = validator.ValidateAtIndexAgainstDims(index, t.dims)
	if err != nil {
		return value, fmt.Errorf("At input index validation failed: %w", err)
	}

	value = t.dataAt(index).(float64)

	return value, nil
}

func (t *CPUTensor) Slice(index []tensor.Range) (o tensor.Tensor, err error) {
	err = validator.ValidateSliceIndexAgainstDims(index, t.dims)
	if err != nil {
		return o, fmt.Errorf("Slice input index validation failed: %w", err)
	}

	r := t.slice(index)
	r.gctx = gradtrack.Slice(r, t, index)

	return r, nil
}

func (t *CPUTensor) Patch(index []tensor.Range, u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("Patch tensors' device validation failed: %w", err)
	}

	err = validator.ValidatePatchIndexAgainstDims(index, _u.dims, t.dims)
	if err != nil {
		return o, fmt.Errorf("Patch input index or tensors' dimension validation failed: %w", err)
	}

	r := t.patch(index, _u)
	r.gctx = gradtrack.Patch(r, t, _u, index)

	return r, nil
}

func (t *CPUTensor) Transpose() (o tensor.Tensor, err error) {
	err = validator.ValidateTransposeDims(t.dims)
	if err != nil {
		return o, fmt.Errorf("Transpose tensor's dimension validation failed: %w", err)
	}

	r := t.transpose()
	r.gctx = gradtrack.Transpose(r, t)

	return r, nil
}

func (t *CPUTensor) Reshape(shape []int) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims(shape)
	if err != nil {
		return o, fmt.Errorf("Reshape input shape validation failed: %w", err)
	}

	err = validator.ValidateReshapeSourceDimsAgainstTargetDims(t.dims, shape)
	if err != nil {
		return o, fmt.Errorf("Reshape input shape validation failed: %w", err)
	}

	r := t.reshape(shape)
	r.gctx = gradtrack.Reshape(r, t)

	return r, nil
}

func (t *CPUTensor) UnSqueeze(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateUnSqueezeDimAgainstDims(dim, t.dims)
	if err != nil {
		return o, fmt.Errorf("UnSqueeze input dimension validation failed: %w", err)
	}

	r := t.unsqueeze(dim)
	r.gctx = gradtrack.UnSqueeze(r, t)

	return r, nil
}

func (t *CPUTensor) Squeeze(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateSqueezeDimAgainstDims(dim, t.dims)
	if err != nil {
		return o, fmt.Errorf("Squeeze input dimension validation failed: %w", err)
	}

	r := t.squeeze(dim)
	r.gctx = gradtrack.Squeeze(r, t)

	return r, nil
}

func (t *CPUTensor) Flatten(fromDim int) (o tensor.Tensor, err error) {
	err = validator.ValidateFlattenDimAgainstDims(fromDim, t.dims)
	if err != nil {
		return o, fmt.Errorf("Flatten input dimension validation failed: %w", err)
	}

	r := t.flatten(fromDim)
	r.gctx = gradtrack.Flatten(r, t)

	return r, nil
}

func (t *CPUTensor) Broadcast(shape []int) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims(shape)
	if err != nil {
		return o, fmt.Errorf("Broadcast input shape validation failed: %w", err)
	}

	err = validator.ValidateBroadcastSourceDimsAgainstTargetDims(t.dims, shape)
	if err != nil {
		return o, fmt.Errorf("Broadcast input shape validation failed: %w", err)
	}

	r := t.broadcast(shape)
	r.gctx = gradtrack.Broadcast(r, t)

	return r, nil
}

func (t *CPUTensor) Sum() float64 {
	return t.sum()
}

func (t *CPUTensor) Max() float64 {
	return t.max(unwrapValue)
}

func (t *CPUTensor) Min() float64 {
	return t.min(unwrapValue)
}

func (t *CPUTensor) Avg() float64 {
	return t.avg()
}

func (t *CPUTensor) Var() float64 {
	return t._var()
}

func (t *CPUTensor) Std() float64 {
	return t.std()
}

func (t *CPUTensor) Mean() float64 {
	return t.mean()
}

func (t *CPUTensor) Argmax(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		return o, fmt.Errorf("Argmax input dimension validation failed: %w", err)
	}

	o = t.argmax(dim)

	return o, nil
}

func (t *CPUTensor) Argmin(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		return o, fmt.Errorf("Argmin input dimension validation failed: %w", err)
	}

	o = t.argmin(dim)

	return o, nil
}

func (t *CPUTensor) SumAlong(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		return o, fmt.Errorf("SumAlong input dimension validation failed: %w", err)
	}

	r := t.sumAlong(dim)
	r.gctx = gradtrack.SumAlong(r, t, dim)

	return r, nil
}

func (t *CPUTensor) MaxAlong(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		return o, fmt.Errorf("MaxAlong input dimension validation failed: %w", err)
	}

	r := t.maxAlong(dim)
	r.gctx = gradtrack.MaxAlong(r, t, dim)

	return r, nil
}

func (t *CPUTensor) MinAlong(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		return o, fmt.Errorf("MinAlong input dimension validation failed: %w", err)
	}

	r := t.minAlong(dim)
	r.gctx = gradtrack.MinAlong(r, t, dim)

	return r, nil
}

func (t *CPUTensor) AvgAlong(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		return o, fmt.Errorf("AvgAlong input dimension validation failed: %w", err)
	}

	r := t.avgAlong(dim)
	r.gctx = gradtrack.AvgAlong(r, t, dim)

	return r, nil
}

func (t *CPUTensor) VarAlong(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		return o, fmt.Errorf("VarAlong input dimension validation failed: %w", err)
	}

	r := t.varAlong(dim)
	r.gctx = gradtrack.VarAlong(r, t, dim)

	return r, nil
}

func (t *CPUTensor) StdAlong(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		return o, fmt.Errorf("StdAlong input dimension validation failed: %w", err)
	}

	r := t.stdAlong(dim)
	r.gctx = gradtrack.StdAlong(r, t, dim)

	return r, nil
}

func (t *CPUTensor) MeanAlong(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		return o, fmt.Errorf("MeanAlong input dimension validation failed: %w", err)
	}

	r := t.meanAlong(dim)
	r.gctx = gradtrack.MeanAlong(r, t, dim)

	return r, nil
}

func (t *CPUTensor) Scale(u float64) (o tensor.Tensor) {
	r := t.scale(u)
	r.gctx = gradtrack.Scale(r, t, u)
	return r
}

func (t *CPUTensor) Pow(u float64) (o tensor.Tensor) {
	r := t.pow(u)
	r.gctx = gradtrack.Pow(r, t, u)
	return r
}

func (t *CPUTensor) Exp() (o tensor.Tensor) {
	r := t.exp()
	r.gctx = gradtrack.Exp(r, t)
	return r
}

func (t *CPUTensor) Log() (o tensor.Tensor) {
	r := t.log()
	r.gctx = gradtrack.Log(r, t)
	return r
}

func (t *CPUTensor) Sin() (o tensor.Tensor) {
	r := t.sin()
	r.gctx = gradtrack.Sin(r, t)
	return r
}

func (t *CPUTensor) Cos() (o tensor.Tensor) {
	r := t.cos()
	r.gctx = gradtrack.Cos(r, t)
	return r
}

func (t *CPUTensor) Tan() (o tensor.Tensor) {
	r := t.tan()
	r.gctx = gradtrack.Tan(r, t)
	return r
}

func (t *CPUTensor) Sinh() (o tensor.Tensor) {
	r := t.sinh()
	r.gctx = gradtrack.Sinh(r, t)
	return r
}

func (t *CPUTensor) Cosh() (o tensor.Tensor) {
	r := t.cosh()
	r.gctx = gradtrack.Cosh(r, t)
	return r
}

func (t *CPUTensor) Tanh() (o tensor.Tensor) {
	r := t.tanh()
	r.gctx = gradtrack.Tanh(r, t)
	return r
}

func (t *CPUTensor) Eq(u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("Eq tensors' device validation failed: %w", err)
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, _u.dims)
	if err != nil {
		return o, fmt.Errorf("Eq tensors' dimension validation failed: %w", err)
	}

	r := t.eq(_u)
	r.gctx = gradtrack.NewGradContext(false)

	return r, nil
}

func (t *CPUTensor) Ne(u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("Ne tensors' device validation failed: %w", err)
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, _u.dims)
	if err != nil {
		return o, fmt.Errorf("Ne tensors' dimension validation failed: %w", err)
	}

	r := t.ne(_u)
	r.gctx = gradtrack.NewGradContext(false)

	return r, nil
}

func (t *CPUTensor) Gt(u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("Gt tensors' device validation failed: %w", err)
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, _u.dims)
	if err != nil {
		return o, fmt.Errorf("Gt tensors' dimension validation failed: %w", err)
	}

	r := t.gt(_u)
	r.gctx = gradtrack.NewGradContext(false)

	return r, nil
}

func (t *CPUTensor) Ge(u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("Ge tensors' device validation failed: %w", err)
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, _u.dims)
	if err != nil {
		return o, fmt.Errorf("Ge tensors' dimension validation failed: %w", err)
	}

	r := t.ge(_u)
	r.gctx = gradtrack.NewGradContext(false)

	return r, nil
}

func (t *CPUTensor) Lt(u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("Lt tensors' device validation failed: %w", err)
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, _u.dims)
	if err != nil {
		return o, fmt.Errorf("Lt tensors' dimension validation failed: %w", err)
	}

	r := t.lt(_u)
	r.gctx = gradtrack.NewGradContext(false)

	return r, nil
}

func (t *CPUTensor) Le(u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("Le tensors' device validation failed: %w", err)
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, _u.dims)
	if err != nil {
		return o, fmt.Errorf("Le tensors' dimension validation failed: %w", err)
	}

	r := t.le(_u)
	r.gctx = gradtrack.NewGradContext(false)

	return r, nil
}

func (t *CPUTensor) ElMax(u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("ElMax tensors' device validation failed: %w", err)
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, _u.dims)
	if err != nil {
		return o, fmt.Errorf("ElMax tensors' dimension validation failed: %w", err)
	}

	r := t.elmax(_u)
	r.gctx = gradtrack.ElMax(r, t, _u)

	return r, nil
}

func (t *CPUTensor) ElMin(u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("ElMin tensors' device validation failed: %w", err)
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, _u.dims)
	if err != nil {
		return o, fmt.Errorf("ElMin tensors' dimension validation failed: %w", err)
	}

	r := t.elmin(_u)
	r.gctx = gradtrack.ElMin(r, t, _u)

	return r, nil
}

func (t *CPUTensor) Add(u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("Add tensors' device validation failed: %w", err)
	}

	t1, t2, err := util.BroadcastForBinaryOps(t, _u)
	if err != nil {
		return o, fmt.Errorf("Add tensors' broadcasting failed: %w", err)
	}

	_t1 := t1.(*CPUTensor)
	_t2 := t2.(*CPUTensor)

	r := _t1.add(_t2)
	r.gctx = gradtrack.Add(r, _t1, _t2)

	return r, nil
}

func (t *CPUTensor) Sub(u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("Sub tensors' device validation failed: %w", err)
	}

	t1, t2, err := util.BroadcastForBinaryOps(t, _u)
	if err != nil {
		return o, fmt.Errorf("Sub tensors' broadcasting failed: %w", err)
	}

	_t1 := t1.(*CPUTensor)
	_t2 := t2.(*CPUTensor)

	r := _t1.sub(_t2)
	r.gctx = gradtrack.Sub(r, _t1, _t2)

	return r, nil
}

func (t *CPUTensor) Mul(u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("Mul tensors' device validation failed: %w", err)
	}

	t1, t2, err := util.BroadcastForBinaryOps(t, _u)
	if err != nil {
		return o, fmt.Errorf("Mul tensors' broadcasting failed: %w", err)
	}

	_t1 := t1.(*CPUTensor)
	_t2 := t2.(*CPUTensor)

	r := _t1.mul(_t2)
	r.gctx = gradtrack.Mul(r, _t1, _t2)

	return r, nil
}

func (t *CPUTensor) Div(u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("Div tensors' device validation failed: %w", err)
	}

	t1, t2, err := util.BroadcastForBinaryOps(t, _u)
	if err != nil {
		return o, fmt.Errorf("Div tensors' broadcasting failed: %w", err)
	}

	_t1 := t1.(*CPUTensor)
	_t2 := t2.(*CPUTensor)

	r := _t1.div(_t2)
	r.gctx = gradtrack.Div(r, _t1, _t2)

	return r, nil
}

func (t *CPUTensor) Dot(u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("Dot tensors' device validation failed: %w", err)
	}

	err = validator.ValidateDotProductDims(t.dims, _u.dims)
	if err != nil {
		return o, fmt.Errorf("Dot tensors' dimension validation failed: %w", err)
	}

	t1, t2, err := util.BroadcastForBinaryOps(t, _u)
	if err != nil {
		return o, fmt.Errorf("Dot tensors' broadcasting failed: %w", err)
	}

	_t1 := t1.(*CPUTensor)
	_t2 := t2.(*CPUTensor)

	r := _t1.dot(_t2)
	r.gctx = gradtrack.Dot(r, _t1, _t2)

	return r, nil
}

func (t *CPUTensor) MatMul(u tensor.Tensor) (o tensor.Tensor, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return o, fmt.Errorf("MatMul tensors' device validation failed: %w", err)
	}

	err = validator.ValidateMatMulDims(t.dims, _u.dims)
	if err != nil {
		return o, fmt.Errorf("MatMul tensors' dimension validation failed: %w", err)
	}

	t1, t2, err := util.BroadcastForMatMul(t, _u)
	if err != nil {
		return o, fmt.Errorf("MatMul tensors' broadcasting failed: %w", err)
	}

	_t1 := t1.(*CPUTensor)
	_t2 := t2.(*CPUTensor)

	r := _t1.matMul(_t2)
	r.gctx = gradtrack.MatMul(r, _t1, _t2)

	return r, nil
}

func (t *CPUTensor) Equals(u tensor.Tensor) (are bool, err error) {
	_u, err := assertCPUTensor(u)
	if err != nil {
		return are, fmt.Errorf("Equals tensors' device validation failed: %w", err)
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, _u.dims)
	if err != nil {
		return are, fmt.Errorf("Equals tensors' dimension validation failed: %w", err)
	}

	are = t.equals(_u)

	return are, nil
}

func (t *CPUTensor) Gradient() tensor.Tensor {
	return t.gctx.Gradient()
}

func (t *CPUTensor) GradientTracked() bool {
	return t.gctx.Tracked()
}

func (t *CPUTensor) ResetGradContext(tracked bool) {
	t.gctx = gradtrack.NewGradContext(tracked)
}

func (t *CPUTensor) GradContext() any {
	return t.gctx
}

func (t *CPUTensor) Device() tensor.Device {
	return tensor.CPU
}
