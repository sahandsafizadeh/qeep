package cputensor

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
	"github.com/sahandsafizadeh/qeep/tensor/internal/validator"
)

/*------------- initializers ------------*/

func Full(value float64, dims []int32, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims(dims)
	if err != nil {
		err = fmt.Errorf("input dimension validation failed: %w", err)
		return
	}

	r := constTensor(value, dims)

	if withGrad {
		r.gctx = gradtrack.Root()
	}

	return r, nil
}

func Zeros(dims []int32, withGrad bool) (o tensor.Tensor, err error) {
	return Full(0., dims, withGrad)
}

func Ones(dims []int32, withGrad bool) (o tensor.Tensor, err error) {
	return Full(1., dims, withGrad)
}

func Eye(n int32, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims([]int32{n})
	if err != nil {
		err = fmt.Errorf("input dimension validation failed: %w", err)
		return
	}

	r := eyeMatrix(n)

	if withGrad {
		r.gctx = gradtrack.Root()
	}

	return r, nil
}

func RandU(l, u float64, dims []int32, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateRandUParams(l, u)
	if err != nil {
		err = fmt.Errorf("random parameter validation failed: %w", err)
		return
	}

	err = validator.ValidateInputDims(dims)
	if err != nil {
		err = fmt.Errorf("input dimension validation failed: %w", err)
		return
	}

	r := uniformRandomTensor(l, u, dims)

	if withGrad {
		r.gctx = gradtrack.Root()
	}

	return r, nil
}

func RandN(u, s float64, dims []int32, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateRandNParams(u, s)
	if err != nil {
		err = fmt.Errorf("random parameter validation failed: %w", err)
		return
	}

	err = validator.ValidateInputDims(dims)
	if err != nil {
		err = fmt.Errorf("input dimension validation failed: %w", err)
		return
	}

	r := normalRandomTensor(u, s, dims)

	if withGrad {
		r.gctx = gradtrack.Root()
	}

	return r, nil
}

func TensorOf(data any, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDataDimUnity(data)
	if err != nil {
		err = fmt.Errorf("input data validation failed: %w", err)
		return
	}

	r := initTensorFromData(data)

	if withGrad {
		r.gctx = gradtrack.Root()
	}

	return r, nil
}

func Concat(ts []tensor.Tensor, dim int32) (o tensor.Tensor, err error) {
	cus, err := assertCPUTensors(ts)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	cusDims := make([][]int32, len(ts))
	for i, cu := range cus {
		cusDims[i] = cu.dims
	}

	err = validator.ValidateConcatTensorsDimsAlongDim(cusDims, dim)
	if err != nil {
		err = fmt.Errorf("inputs' dimension validation failed: %w", err)
		return
	}

	r := initConcatResultTensor(cus, dim)

	if gradtrack.ForbiddenForAny(ts...) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(ts...) {
		r.gctx = gradtrack.Concat(r, ts, dim)
	}

	return r, nil
}

/*--------------- methods ---------------*/

func (t *CPUTensor) NElems() (n int64) {
	return t.numElems()
}

func (t *CPUTensor) Shape() (shape []int32) {
	shape = make([]int32, len(t.dims))
	copy(shape, t.dims)
	return shape
}

func (t *CPUTensor) At(index ...int32) (value float64, err error) {
	err = validator.ValidateAtIndexAgainstDims(index, t.dims)
	if err != nil {
		err = fmt.Errorf("input index validation failed: %w", err)
		return
	}

	return t.dataAt(index).(float64), nil
}

func (t *CPUTensor) Slice(index []tensor.Range) (o tensor.Tensor, err error) {
	err = validator.ValidateSliceIndexAgainstDims(index, t.dims)
	if err != nil {
		err = fmt.Errorf("input index validation failed: %w", err)
		return
	}

	r := t.slice(index)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Slice(r, t, index)
	}

	return r, nil
}

func (t *CPUTensor) Patch(index []tensor.Range, u tensor.Tensor) (o tensor.Tensor, err error) {
	cu, err := assertCPUTensor(u)
	if err != nil {
		err = fmt.Errorf("tensors' device validation failed: %w", err)
		return
	}

	err = validator.ValidatePatchIndexAgainstDims(index, cu.dims, t.dims)
	if err != nil {
		err = fmt.Errorf("input index or tensors' dimension validation failed: %w", err)
		return
	}

	r := t.patch(index, cu)

	if gradtrack.ForbiddenForAny(t, u) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t, u) {
		r.gctx = gradtrack.Patch(r, t, u, index)
	}

	return r, nil
}

func (t *CPUTensor) Transpose() (o tensor.Tensor, err error) {
	err = validator.ValidateTransposeDims(t.dims)
	if err != nil {
		err = fmt.Errorf("tensor's dimension validation failed: %w", err)
		return
	}

	r := t.transpose()

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Transpose(r, t)
	}

	return r, nil
}

func (t *CPUTensor) Reshape(shape ...int32) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims(shape)
	if err != nil {
		err = fmt.Errorf("input shape validation failed: %w", err)
		return
	}

	err = validator.ValidateReshapeSourceDimsAgainstTargetDims(t.dims, shape)
	if err != nil {
		err = fmt.Errorf("input shape validation failed: %w", err)
		return
	}

	r := t.reshape(shape)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Reshape(r, t)
	}

	return r, nil
}

func (t *CPUTensor) UnSqueeze(dim int32) (o tensor.Tensor, err error) {
	err = validator.ValidateUnSqueezeDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("input dimension validation failed: %w", err)
		return
	}

	r := t.unSqueeze(dim)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.UnSqueeze(r, t)
	}

	return r, nil
}

func (t *CPUTensor) Squeeze(dim int32) (o tensor.Tensor, err error) {
	err = validator.ValidateSqueezeDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("input dimension validation failed: %w", err)
		return
	}

	r := t.squeeze(dim)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Squeeze(r, t)
	}

	return r, nil
}

func (t *CPUTensor) Flatten(fromDim int32) (o tensor.Tensor, err error) {
	err = validator.ValidateFlattenDimAgainstDims(fromDim, t.dims)
	if err != nil {
		err = fmt.Errorf("input dimension validation failed: %w", err)
		return
	}

	r := t.flatten(fromDim)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Flatten(r, t)
	}

	return r, nil
}

func (t *CPUTensor) Broadcast(shape ...int32) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims(shape)
	if err != nil {
		err = fmt.Errorf("input shape validation failed: %w", err)
		return
	}

	err = validator.ValidateBroadcastSourceDimsAgainstTargetDims(t.dims, shape)
	if err != nil {
		err = fmt.Errorf("input shape validation failed: %w", err)
		return
	}

	r := t.broadcast(shape)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Broadcast(r, t)
	}

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

func (t *CPUTensor) SumAlong(dim int32) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("input dimension validation failed: %w", err)
		return
	}

	r := t.sumAlong(dim)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.SumAlong(r, t, dim)
	}

	return r, nil
}

func (t *CPUTensor) MaxAlong(dim int32) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("input dimension validation failed: %w", err)
		return
	}

	r := t.maxAlong(dim)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.MaxAlong(r, t, dim)
	}

	return r, nil
}

func (t *CPUTensor) MinAlong(dim int32) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("input dimension validation failed: %w", err)
		return
	}

	r := t.minAlong(dim)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.MinAlong(r, t, dim)
	}

	return r, nil
}

func (t *CPUTensor) AvgAlong(dim int32) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("input dimension validation failed: %w", err)
		return
	}

	r := t.avgAlong(dim)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.AvgAlong(r, t, dim)
	}

	return r, nil
}

func (t *CPUTensor) VarAlong(dim int32) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("input dimension validation failed: %w", err)
		return
	}

	r := t.varAlong(dim)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.VarAlong(r, t, dim)
	}

	return r, nil
}

func (t *CPUTensor) StdAlong(dim int32) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("input dimension validation failed: %w", err)
		return
	}

	r := t.stdAlong(dim)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.StdAlong(r, t, dim)
	}

	return r, nil
}

func (t *CPUTensor) MeanAlong(dim int32) (o tensor.Tensor, err error) {
	err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	if err != nil {
		err = fmt.Errorf("input dimension validation failed: %w", err)
		return
	}

	r := t.meanAlong(dim)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.MeanAlong(r, t, dim)
	}

	return r, nil
}

func (t *CPUTensor) Scale(u float64) (o tensor.Tensor) {
	r := t.scale(u)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Scale(r, t, u)
	}

	return r
}

func (t *CPUTensor) Pow(u float64) (o tensor.Tensor) {
	r := t.pow(u)

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Pow(r, t, u)
	}

	return r
}

func (t *CPUTensor) Exp() (o tensor.Tensor) {
	r := t.exp()

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Exp(r, t)
	}

	return r
}

func (t *CPUTensor) Log() (o tensor.Tensor) {
	r := t.log()

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Log(r, t)
	}

	return r
}

func (t *CPUTensor) Sin() (o tensor.Tensor) {
	r := t.sin()

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Sin(r, t)
	}

	return r
}

func (t *CPUTensor) Cos() (o tensor.Tensor) {
	r := t.cos()

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Cos(r, t)
	}

	return r
}

func (t *CPUTensor) Tan() (o tensor.Tensor) {
	r := t.tan()

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Tan(r, t)
	}

	return r
}

func (t *CPUTensor) Sinh() (o tensor.Tensor) {
	r := t.sinh()

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Sinh(r, t)
	}

	return r
}

func (t *CPUTensor) Cosh() (o tensor.Tensor) {
	r := t.cosh()

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Cosh(r, t)
	}

	return r
}

func (t *CPUTensor) Tanh() (o tensor.Tensor) {
	r := t.tanh()

	if gradtrack.ForbiddenForAny(t) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(t) {
		r.gctx = gradtrack.Tanh(r, t)
	}

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

	return t.eq(cu), nil
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

	return t.ne(cu), nil
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

	return t.gt(cu), nil
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

	return t.ge(cu), nil
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

	return t.lt(cu), nil
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

	return t.le(cu), nil
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

	if gradtrack.ForbiddenForAny(ct1, ct2) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(ct1, ct2) {
		r.gctx = gradtrack.Add(r, ct1, ct2)
	}

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

	if gradtrack.ForbiddenForAny(ct1, ct2) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(ct1, ct2) {
		r.gctx = gradtrack.Sub(r, ct1, ct2)
	}

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

	if gradtrack.ForbiddenForAny(ct1, ct2) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(ct1, ct2) {
		r.gctx = gradtrack.Mul(r, ct1, ct2)
	}

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

	if gradtrack.ForbiddenForAny(ct1, ct2) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(ct1, ct2) {
		r.gctx = gradtrack.Div(r, ct1, ct2)
	}

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

	if gradtrack.ForbiddenForAny(ct1, ct2) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(ct1, ct2) {
		r.gctx = gradtrack.Dot(r, ct1, ct2)
	}

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

	if gradtrack.ForbiddenForAny(ct1, ct2) {
		r.gctx = gradtrack.Forbidden()
	} else if gradtrack.RequiredForAny(ct1, ct2) {
		r.gctx = gradtrack.MatMul(r, ct1, ct2)
	}

	return r, nil
}

func (t *CPUTensor) Equals(u tensor.Tensor) (are bool, err error) {
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

	return t.equals(cu), nil
}

func (t *CPUTensor) GradContext() (gctx any) {
	return t.gctx
}

func (t *CPUTensor) Gradient() (g tensor.Tensor) {
	return t.gctx.Grad()
}

func (t *CPUTensor) Detach() (o tensor.Tensor) {
	return t.slice(nil)
}
