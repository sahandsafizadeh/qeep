package cputensor

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/validator"
)

/*------------- initializers ------------*/

func Full(dims []int, value float64, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims(dims)
	if err != nil {
		err = fmt.Errorf("Full input dimension validation failed: %w", err)
		return
	}

	r := constTensor(value, dims)
	r.gctx = gradtrack.NewGradContext(withGrad)

	return r, nil
}

func Zeros(dims []int, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims(dims)
	if err != nil {
		err = fmt.Errorf("Zeros input dimension validation failed: %w", err)
		return
	}

	return Full(dims, 0., withGrad)
}

func Ones(dims []int, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims(dims)
	if err != nil {
		err = fmt.Errorf("Ones input dimension validation failed: %w", err)
		return
	}

	return Full(dims, 1., withGrad)
}

func Eye(n int, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims([]int{n, n})
	if err != nil {
		err = fmt.Errorf("Eye input dimension validation failed: %w", err)
		return
	}

	r := eyeMatrix(n)
	r.gctx = gradtrack.NewGradContext(withGrad)

	return r, nil
}

func RandU(dims []int, l, u float64, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateRandUParams(l, u)
	if err != nil {
		err = fmt.Errorf("RandU random parameter validation failed: %w", err)
		return
	}

	err = validator.ValidateInputDims(dims)
	if err != nil {
		err = fmt.Errorf("RandU input dimension validation failed: %w", err)
		return
	}

	r := uniformRandomTensor(l, u, dims)
	r.gctx = gradtrack.NewGradContext(withGrad)

	return r, nil
}

func RandN(dims []int, u, s float64, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateRandNParams(u, s)
	if err != nil {
		err = fmt.Errorf("RandN random parameter validation failed: %w", err)
		return
	}

	err = validator.ValidateInputDims(dims)
	if err != nil {
		err = fmt.Errorf("RandN input dimension validation failed: %w", err)
		return
	}

	r := normalRandomTensor(u, s, dims)
	r.gctx = gradtrack.NewGradContext(withGrad)

	return r, nil
}

func TensorOf(data any, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDataDimUnity(data)
	if err != nil {
		err = fmt.Errorf("TensorOf input data validation failed: %w", err)
		return
	}

	r := initTensorFromData(data)
	r.gctx = gradtrack.NewGradContext(withGrad)

	return r, nil
}

func Concat(ts []tensor.Tensor, dim int) (o tensor.Tensor, err error) {
	cus, err := assertCPUTensors(ts)
	if err != nil {
		err = fmt.Errorf("Concat tensors' device validation failed: %w", err)
		return
	}

	cusDims := make([][]int, len(ts))
	for i, cu := range cus {
		cusDims[i] = cu.dims
	}

	err = validator.ValidateConcatTensorsDimsAlongDim(cusDims, dim)
	if err != nil {
		err = fmt.Errorf("Concat inputs' dimension validation failed: %w", err)
		return
	}

	r := initConcatResultTensor(cus, dim)
	r.gctx = gradtrack.Concat(r, ts, dim)

	return r, nil
}

/*--------------- methods ---------------*/

func (t *CPUTensor) NElems() (n int) {
	return t.numElems()
}

func (t *CPUTensor) Shape() (shape []int) {
	shape = make([]int, len(t.dims))
	copy(shape, t.dims)
	return shape
}

func (t *CPUTensor) At(index ...int) (value float64, err error) {
	err = validator.ValidateAtIndexAgainstDims(index, t.dims)
	if err != nil {
		err = fmt.Errorf("At input index validation failed: %w", err)
		return
	}

	return t.dataAt(index).(float64), nil
}

func (t *CPUTensor) Slice(index []tensor.Range) (o tensor.Tensor, err error) {
	err = validator.ValidateSliceIndexAgainstDims(index, t.dims)
	if err != nil {
		err = fmt.Errorf("Slice input index validation failed: %w", err)
		return
	}

	r := t.slice(index)
	r.gctx = gradtrack.Slice(r, t, index)

	return r, nil
}

func (t *CPUTensor) Patch(index []tensor.Range, u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("Patch tensors' device validation failed: %w", err)
		return
	}

	err = validator.ValidatePatchIndexAgainstDims(index, cu.dims, t.dims)
	if err != nil {
		err = fmt.Errorf("Patch input index or tensors' dimension validation failed: %w", err)
		return
	}

	r := t.patch(index, cu)
	r.gctx = gradtrack.Patch(r, t, u, index)

	return r, nil
}

func (t *CPUTensor) Transpose() (o tensor.Tensor, err error) {
	err = validator.ValidateTransposeDims(t.dims)
	if err != nil {
		err = fmt.Errorf("Transpose tensor's dimension validation failed: %w", err)
		return
	}

	r := t.transpose()
	r.gctx = gradtrack.Transpose(r, t)

	return r, nil
}

func (t *CPUTensor) Reshape(shape []int) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims(shape)
	if err != nil {
		err = fmt.Errorf("Reshape input shape validation failed: %w", err)
		return
	}

	err = validator.ValidateReshapeSourceDimsAgainstTargetDims(t.dims, shape)
	if err != nil {
		err = fmt.Errorf("Reshape input shape validation failed: %w", err)
		return
	}

	r := t.reshape(shape)
	r.gctx = gradtrack.Reshape(r, t)

	return r, nil
}

func (t *CPUTensor) UnSqueeze(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateUnSqueezeDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("UnSqueeze input dimension validation failed: %w", err)
		return
	}

	r := t.unSqueeze(dim)
	r.gctx = gradtrack.UnSqueeze(r, t)

	return r, nil
}

func (t *CPUTensor) Squeeze(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateSqueezeDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("Squeeze input dimension validation failed: %w", err)
		return
	}

	r := t.squeeze(dim)
	r.gctx = gradtrack.Squeeze(r, t)

	return r, nil
}

func (t *CPUTensor) Flatten(fromDim int) (o tensor.Tensor, err error) {
	err = validator.ValidateFlattenDimAgainstDims(fromDim, t.dims)
	if err != nil {
		err = fmt.Errorf("Flatten input dimension validation failed: %w", err)
		return
	}

	r := t.flatten(fromDim)
	r.gctx = gradtrack.Flatten(r, t)

	return r, nil
}

func (t *CPUTensor) Broadcast(shape []int) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims(shape)
	if err != nil {
		err = fmt.Errorf("Broadcast input shape validation failed: %w", err)
		return
	}

	err = validator.ValidateBroadcastSourceDimsAgainstTargetDims(t.dims, shape)
	if err != nil {
		err = fmt.Errorf("Broadcast input shape validation failed: %w", err)
		return
	}

	r := t.broadcast(shape)
	r.gctx = gradtrack.Broadcast(r, t)

	return r, nil
}

func (t *CPUTensor) Sum() (value float64) {
	return t.sum()
}

func (t *CPUTensor) Max() (value float64) {
	return t.max()
}

func (t *CPUTensor) Min() (value float64) {
	return t.min()
}

func (t *CPUTensor) Avg() (value float64) {
	return t.avg()
}

func (t *CPUTensor) Var() (value float64) {
	return t._var()
}

func (t *CPUTensor) Std() (value float64) {
	return t.std()
}

func (t *CPUTensor) Mean() (value float64) {
	return t.mean()
}

func (t *CPUTensor) SumAlong(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("SumAlong input dimension validation failed: %w", err)
		return
	}

	r := t.sumAlong(dim)
	r.gctx = gradtrack.SumAlong(r, t, dim)

	return r, nil
}

func (t *CPUTensor) MaxAlong(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("MaxAlong input dimension validation failed: %w", err)
		return
	}

	r := t.maxAlong(dim)
	r.gctx = gradtrack.MaxAlong(r, t, dim)

	return r, nil
}

func (t *CPUTensor) MinAlong(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("MinAlong input dimension validation failed: %w", err)
		return
	}

	r := t.minAlong(dim)
	r.gctx = gradtrack.MinAlong(r, t, dim)

	return r, nil
}

func (t *CPUTensor) AvgAlong(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("AvgAlong input dimension validation failed: %w", err)
		return
	}

	r := t.avgAlong(dim)
	r.gctx = gradtrack.AvgAlong(r, t, dim)

	return r, nil
}

func (t *CPUTensor) VarAlong(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("VarAlong input dimension validation failed: %w", err)
		return
	}

	r := t.varAlong(dim)
	r.gctx = gradtrack.VarAlong(r, t, dim)

	return r, nil
}

func (t *CPUTensor) StdAlong(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("StdAlong input dimension validation failed: %w", err)
		return
	}

	r := t.stdAlong(dim)
	r.gctx = gradtrack.StdAlong(r, t, dim)

	return r, nil
}

func (t *CPUTensor) MeanAlong(dim int) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("MeanAlong input dimension validation failed: %w", err)
		return
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
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	if err != nil {
		err = fmt.Errorf("tensors' dimension validation failed: %w", err)
		return
	}

	r := t.eq(cu)
	r.gctx = gradtrack.NewGradContext(false)

	return r, nil
}

func (t *CPUTensor) Ne(u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	if err != nil {
		err = fmt.Errorf("tensors' dimension validation failed: %w", err)
		return
	}

	r := t.ne(cu)
	r.gctx = gradtrack.NewGradContext(false)

	return r, nil
}

func (t *CPUTensor) Gt(u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	if err != nil {
		err = fmt.Errorf("tensors' dimension validation failed: %w", err)
		return
	}

	r := t.gt(cu)
	r.gctx = gradtrack.NewGradContext(false)

	return r, nil
}

func (t *CPUTensor) Ge(u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	if err != nil {
		err = fmt.Errorf("tensors' dimension validation failed: %w", err)
		return
	}

	r := t.ge(cu)
	r.gctx = gradtrack.NewGradContext(false)

	return r, nil
}

func (t *CPUTensor) Lt(u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	if err != nil {
		err = fmt.Errorf("tensors' dimension validation failed: %w", err)
		return
	}

	r := t.lt(cu)
	r.gctx = gradtrack.NewGradContext(false)

	return r, nil
}

func (t *CPUTensor) Le(u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	if err != nil {
		err = fmt.Errorf("tensors' dimension validation failed: %w", err)
		return
	}

	r := t.le(cu)
	r.gctx = gradtrack.NewGradContext(false)

	return r, nil
}

func (t *CPUTensor) ElMax(u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	if err != nil {
		err = fmt.Errorf("tensors' dimension validation failed: %w", err)
		return
	}

	r := t.elmax(cu)
	r.gctx = gradtrack.ElMax(r, t, cu)

	return r, nil
}

func (t *CPUTensor) ElMin(u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	if err != nil {
		err = fmt.Errorf("tensors' dimension validation failed: %w", err)
		return
	}

	r := t.elmin(cu)
	r.gctx = gradtrack.ElMin(r, t, cu)

	return r, nil
}

func (t *CPUTensor) Add(u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	ct1, ct2, err := broadcastForBinaryOp(t, cu)
	if err != nil {
		err = fmt.Errorf("tensors' broadcasting failed: %w", err)
		return
	}

	r := ct1.add(ct2)
	r.gctx = gradtrack.Add(r, ct1, ct2)

	return r, nil
}

func (t *CPUTensor) Sub(u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	ct1, ct2, err := broadcastForBinaryOp(t, cu)
	if err != nil {
		err = fmt.Errorf("tensors' broadcasting failed: %w", err)
		return
	}

	r := ct1.sub(ct2)
	r.gctx = gradtrack.Sub(r, ct1, ct2)

	return r, nil
}

func (t *CPUTensor) Mul(u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	ct1, ct2, err := broadcastForBinaryOp(t, cu)
	if err != nil {
		err = fmt.Errorf("tensors' broadcasting failed: %w", err)
		return
	}

	r := ct1.mul(ct2)
	r.gctx = gradtrack.Mul(r, ct1, ct2)

	return r, nil
}

func (t *CPUTensor) Div(u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	ct1, ct2, err := broadcastForBinaryOp(t, cu)
	if err != nil {
		err = fmt.Errorf("tensors' broadcasting failed: %w", err)
		return
	}

	r := ct1.div(ct2)
	r.gctx = gradtrack.Div(r, ct1, ct2)

	return r, nil
}

func (t *CPUTensor) Dot(u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	err = validator.ValidateDotProductDims(t.dims, cu.dims)
	if err != nil {
		err = fmt.Errorf("tensors' dimension validation failed: %w", err)
		return
	}

	ct1, ct2, err := broadcastForBinaryOp(t, cu)
	if err != nil {
		err = fmt.Errorf("tensors' broadcasting failed: %w", err)
		return
	}

	r := ct1.dot(ct2)
	r.gctx = gradtrack.Dot(r, ct1, ct2)

	return r, nil
}

func (t *CPUTensor) MatMul(u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	err = validator.ValidateMatMulDims(t.dims, cu.dims)
	if err != nil {
		err = fmt.Errorf("tensors' dimension validation failed: %w", err)
		return
	}

	ct1, ct2, err := broadcastForMatMul(t, cu)
	if err != nil {
		err = fmt.Errorf("tensors' broadcasting failed: %w", err)
		return
	}

	r := ct1.matMul(ct2)
	r.gctx = gradtrack.MatMul(r, ct1, ct2)

	return r, nil
}

func (t *CPUTensor) Equals(u tensor.Tensor) (are bool, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("Equals tensors' device validation failed: %w", err)
		return
	}

	err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	if err != nil {
		err = fmt.Errorf("Equals tensors' dimension validation failed: %w", err)
		return
	}

	return t.equals(cu), nil
}

func (t *CPUTensor) GradContext() (gctx any) {
	return t.gctx
}

func (t *CPUTensor) ResetGradContext(tracked bool) {
	t.gctx = gradtrack.NewGradContext(tracked)
}

func (t *CPUTensor) Gradient() (g tensor.Tensor) {
	return t.gctx.Gradient()
}
