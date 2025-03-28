package cudatensor

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"
	"github.com/sahandsafizadeh/qeep/tensor/internal/validator"
)

func Full(dims []int, value float64, withGrad bool) (o tensor.Tensor, err error) {
	err = validator.ValidateInputDims(dims)
	if err != nil {
		err = fmt.Errorf("Full input dimension validation failed: %w", err)
		return
	}

	r := constTensor(dims, value)
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
	return
	// err = validator.ValidateInputDims([]int{n, n})
	// if err != nil {
	// 	err = fmt.Errorf("Eye input dimension validation failed: %w", err)
	// 	return
	// }

	// r := eyeMatrix(n)
	// r.gctx = gradtrack.NewGradContext(withGrad)

	// return r, nil
}

func RandU(dims []int, l, u float64, withGrad bool) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateRandUParams(l, u)
	// if err != nil {
	// 	err = fmt.Errorf("RandU random parameter validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateInputDims(dims)
	// if err != nil {
	// 	err = fmt.Errorf("RandU input dimension validation failed: %w", err)
	// 	return
	// }

	// r := uniformRandomTensor(l, u, dims)
	// r.gctx = gradtrack.NewGradContext(withGrad)

	// return r, nil
}

func RandN(dims []int, u, s float64, withGrad bool) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateRandNParams(u, s)
	// if err != nil {
	// 	err = fmt.Errorf("RandN random parameter validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateInputDims(dims)
	// if err != nil {
	// 	err = fmt.Errorf("RandN input dimension validation failed: %w", err)
	// 	return
	// }

	// r := normalRandomTensor(u, s, dims)
	// r.gctx = gradtrack.NewGradContext(withGrad)

	// return r, nil
}

func TensorOf(data any, withGrad bool) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateInputDataDimUnity(data)
	// if err != nil {
	// 	err = fmt.Errorf("TensorOf input data validation failed: %w", err)
	// 	return
	// }

	// r := initTensorFromData(data)
	// r.gctx = gradtrack.NewGradContext(withGrad)

	// return r, nil
}

func Concat(ts []tensor.Tensor, dim int) (o tensor.Tensor, err error) {
	return
	// cus, err := assertCPUTensors(ts)
	// if err != nil {
	// 	err = fmt.Errorf("Concat tensors' device validation failed: %w", err)
	// 	return
	// }

	// cusDims := make([][]int, len(ts))
	// for i, cu := range cus {
	// 	cusDims[i] = cu.dims
	// }

	// err = validator.ValidateConcatTensorsDimsAlongDim(cusDims, dim)
	// if err != nil {
	// 	err = fmt.Errorf("Concat inputs' dimension validation failed: %w", err)
	// 	return
	// }

	// r := initConcatResultTensor(cus, dim)
	// r.gctx = gradtrack.Concat(r, ts, dim)

	// return r, nil
}

/*--------------- methods ---------------*/

func (t *CUDATensor) NElems() (n int) {
	return t.numElems()
}

func (t *CUDATensor) Shape() (shape []int) {
	return
	// shape = make([]int, len(t.dims))
	// copy(shape, t.dims)
	// return shape
}

func (t *CUDATensor) At(index ...int) (value float64, err error) {
	return
	// err = validator.ValidateAtIndexAgainstDims(index, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("At input index validation failed: %w", err)
	// 	return
	// }

	// return t.dataAt(index).(float64), nil
}

func (t *CUDATensor) Slice(index []tensor.Range) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateSliceIndexAgainstDims(index, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Slice input index validation failed: %w", err)
	// 	return
	// }

	// r := t.slice(index)
	// r.gctx = gradtrack.Slice(r, t, index)

	// return r, nil
}

func (t *CUDATensor) Patch(index []tensor.Range, u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("Patch tensors' device validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidatePatchIndexAgainstDims(index, cu.dims, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Patch input index or tensors' dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.patch(index, cu)
	// r.gctx = gradtrack.Patch(r, t, u, index)

	// return r, nil
}

func (t *CUDATensor) Transpose() (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateTransposeDims(t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Transpose tensor's dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.transpose()
	// r.gctx = gradtrack.Transpose(r, t)

	// return r, nil
}

func (t *CUDATensor) Reshape(shape []int) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateInputDims(shape)
	// if err != nil {
	// 	err = fmt.Errorf("Reshape input shape validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateReshapeSourceDimsAgainstTargetDims(t.dims, shape)
	// if err != nil {
	// 	err = fmt.Errorf("Reshape input shape validation failed: %w", err)
	// 	return
	// }

	// r := t.reshape(shape)
	// r.gctx = gradtrack.Reshape(r, t)

	// return r, nil
}

func (t *CUDATensor) UnSqueeze(dim int) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateUnSqueezeDimAgainstDims(dim, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("UnSqueeze input dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.unSqueeze(dim)
	// r.gctx = gradtrack.UnSqueeze(r, t)

	// return r, nil
}

func (t *CUDATensor) Squeeze(dim int) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateSqueezeDimAgainstDims(dim, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Squeeze input dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.squeeze(dim)
	// r.gctx = gradtrack.Squeeze(r, t)

	// return r, nil
}

func (t *CUDATensor) Flatten(fromDim int) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateFlattenDimAgainstDims(fromDim, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Flatten input dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.flatten(fromDim)
	// r.gctx = gradtrack.Flatten(r, t)

	// return r, nil
}

func (t *CUDATensor) Broadcast(shape []int) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateInputDims(shape)
	// if err != nil {
	// 	err = fmt.Errorf("Broadcast input shape validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateBroadcastSourceDimsAgainstTargetDims(t.dims, shape)
	// if err != nil {
	// 	err = fmt.Errorf("Broadcast input shape validation failed: %w", err)
	// 	return
	// }

	// r := t.broadcast(shape)
	// r.gctx = gradtrack.Broadcast(r, t)

	// return r, nil
}

func (t *CUDATensor) Sum() (value float64) {
	return
	// return t.sum()
}

func (t *CUDATensor) Max() (value float64) {
	return
	// return t.max(unwrapValue)
}

func (t *CUDATensor) Min() (value float64) {
	return
	// return t.min(unwrapValue)
}

func (t *CUDATensor) Avg() (value float64) {
	return
	// return t.avg()
}

func (t *CUDATensor) Var() (value float64) {
	return
	// return t._var()
}

func (t *CUDATensor) Std() (value float64) {
	return
	// return t.std()
}

func (t *CUDATensor) Mean() (value float64) {
	return
	// return t.mean()
}

func (t *CUDATensor) Argmax(dim int) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Argmax input dimension validation failed: %w", err)
	// 	return
	// }

	// return t.argmax(dim), nil
}

func (t *CUDATensor) Argmin(dim int) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Argmin input dimension validation failed: %w", err)
	// 	return
	// }

	// return t.argmin(dim), nil
}

func (t *CUDATensor) SumAlong(dim int) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("SumAlong input dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.sumAlong(dim)
	// r.gctx = gradtrack.SumAlong(r, t, dim)

	// return r, nil
}

func (t *CUDATensor) MaxAlong(dim int) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("MaxAlong input dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.maxAlong(dim)
	// r.gctx = gradtrack.MaxAlong(r, t, dim)

	// return r, nil
}

func (t *CUDATensor) MinAlong(dim int) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("MinAlong input dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.minAlong(dim)
	// r.gctx = gradtrack.MinAlong(r, t, dim)

	// return r, nil
}

func (t *CUDATensor) AvgAlong(dim int) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("AvgAlong input dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.avgAlong(dim)
	// r.gctx = gradtrack.AvgAlong(r, t, dim)

	// return r, nil
}

func (t *CUDATensor) VarAlong(dim int) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("VarAlong input dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.varAlong(dim)
	// r.gctx = gradtrack.VarAlong(r, t, dim)

	// return r, nil
}

func (t *CUDATensor) StdAlong(dim int) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("StdAlong input dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.stdAlong(dim)
	// r.gctx = gradtrack.StdAlong(r, t, dim)

	// return r, nil
}

func (t *CUDATensor) MeanAlong(dim int) (o tensor.Tensor, err error) {
	return
	// err = validator.ValidateReducedDimAgainstDims(dim, t.dims)
	// if err != nil {
	// 	err = fmt.Errorf("MeanAlong input dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.meanAlong(dim)
	// r.gctx = gradtrack.MeanAlong(r, t, dim)

	// return r, nil
}

func (t *CUDATensor) Scale(u float64) (o tensor.Tensor) {
	r := t.scale(u)
	r.gctx = gradtrack.Scale(r, t, u)
	return r
}

func (t *CUDATensor) Pow(u float64) (o tensor.Tensor) {
	r := t.pow(u)
	r.gctx = gradtrack.Pow(r, t, u)
	return r
}

func (t *CUDATensor) Exp() (o tensor.Tensor) {
	r := t.exp()
	r.gctx = gradtrack.Exp(r, t)
	return r
}

func (t *CUDATensor) Log() (o tensor.Tensor) {
	return
	// r := t.log()
	// r.gctx = gradtrack.Log(r, t)
	// return r
}

func (t *CUDATensor) Sin() (o tensor.Tensor) {
	return
	// r := t.sin()
	// r.gctx = gradtrack.Sin(r, t)
	// return r
}

func (t *CUDATensor) Cos() (o tensor.Tensor) {
	return
	// r := t.cos()
	// r.gctx = gradtrack.Cos(r, t)
	// return r
}

func (t *CUDATensor) Tan() (o tensor.Tensor) {
	return
	// r := t.tan()
	// r.gctx = gradtrack.Tan(r, t)
	// return r
}

func (t *CUDATensor) Sinh() (o tensor.Tensor) {
	return
	// r := t.sinh()
	// r.gctx = gradtrack.Sinh(r, t)
	// return r
}

func (t *CUDATensor) Cosh() (o tensor.Tensor) {
	return
	// r := t.cosh()
	// r.gctx = gradtrack.Cosh(r, t)
	// return r
}

func (t *CUDATensor) Tanh() (o tensor.Tensor) {
	return
	// r := t.tanh()
	// r.gctx = gradtrack.Tanh(r, t)
	// return r
}

func (t *CUDATensor) Eq(u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("Eq tensors' device validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Eq tensors' dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.eq(cu)
	// r.gctx = gradtrack.NewGradContext(false)

	// return r, nil
}

func (t *CUDATensor) Ne(u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("Ne tensors' device validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Ne tensors' dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.ne(cu)
	// r.gctx = gradtrack.NewGradContext(false)

	// return r, nil
}

func (t *CUDATensor) Gt(u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("Gt tensors' device validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Gt tensors' dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.gt(cu)
	// r.gctx = gradtrack.NewGradContext(false)

	// return r, nil
}

func (t *CUDATensor) Ge(u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("Ge tensors' device validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Ge tensors' dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.ge(cu)
	// r.gctx = gradtrack.NewGradContext(false)

	// return r, nil
}

func (t *CUDATensor) Lt(u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("Lt tensors' device validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Lt tensors' dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.lt(cu)
	// r.gctx = gradtrack.NewGradContext(false)

	// return r, nil
}

func (t *CUDATensor) Le(u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("Le tensors' device validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Le tensors' dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.le(cu)
	// r.gctx = gradtrack.NewGradContext(false)

	// return r, nil
}

func (t *CUDATensor) ElMax(u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("ElMax tensors' device validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	// if err != nil {
	// 	err = fmt.Errorf("ElMax tensors' dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.elmax(cu)
	// r.gctx = gradtrack.ElMax(r, t, cu)

	// return r, nil
}

func (t *CUDATensor) ElMin(u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("ElMin tensors' device validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	// if err != nil {
	// 	err = fmt.Errorf("ElMin tensors' dimension validation failed: %w", err)
	// 	return
	// }

	// r := t.elmin(cu)
	// r.gctx = gradtrack.ElMin(r, t, cu)

	// return r, nil
}

func (t *CUDATensor) Add(u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("Add tensors' device validation failed: %w", err)
	// 	return
	// }

	// ct1, ct2, err := broadcastForBinaryOp(t, cu)
	// if err != nil {
	// 	err = fmt.Errorf("Add tensors' broadcasting failed: %w", err)
	// 	return
	// }

	// r := ct1.add(ct2)
	// r.gctx = gradtrack.Add(r, ct1, ct2)

	// return r, nil
}

func (t *CUDATensor) Sub(u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("Sub tensors' device validation failed: %w", err)
	// 	return
	// }

	// ct1, ct2, err := broadcastForBinaryOp(t, cu)
	// if err != nil {
	// 	err = fmt.Errorf("Sub tensors' broadcasting failed: %w", err)
	// 	return
	// }

	// r := ct1.sub(ct2)
	// r.gctx = gradtrack.Sub(r, ct1, ct2)

	// return r, nil
}

func (t *CUDATensor) Mul(u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("Mul tensors' device validation failed: %w", err)
	// 	return
	// }

	// ct1, ct2, err := broadcastForBinaryOp(t, cu)
	// if err != nil {
	// 	err = fmt.Errorf("Mul tensors' broadcasting failed: %w", err)
	// 	return
	// }

	// r := ct1.mul(ct2)
	// r.gctx = gradtrack.Mul(r, ct1, ct2)

	// return r, nil
}

func (t *CUDATensor) Div(u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("Div tensors' device validation failed: %w", err)
	// 	return
	// }

	// ct1, ct2, err := broadcastForBinaryOp(t, cu)
	// if err != nil {
	// 	err = fmt.Errorf("Div tensors' broadcasting failed: %w", err)
	// 	return
	// }

	// r := ct1.div(ct2)
	// r.gctx = gradtrack.Div(r, ct1, ct2)

	// return r, nil
}

func (t *CUDATensor) Dot(u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("Dot tensors' device validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateDotProductDims(t.dims, cu.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Dot tensors' dimension validation failed: %w", err)
	// 	return
	// }

	// ct1, ct2, err := broadcastForBinaryOp(t, cu)
	// if err != nil {
	// 	err = fmt.Errorf("Dot tensors' broadcasting failed: %w", err)
	// 	return
	// }

	// r := ct1.dot(ct2)
	// r.gctx = gradtrack.Dot(r, ct1, ct2)

	// return r, nil
}

func (t *CUDATensor) MatMul(u tensor.Tensor) (o tensor.Tensor, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("MatMul tensors' device validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateMatMulDims(t.dims, cu.dims)
	// if err != nil {
	// 	err = fmt.Errorf("MatMul tensors' dimension validation failed: %w", err)
	// 	return
	// }

	// ct1, ct2, err := broadcastForMatMul(t, cu)
	// if err != nil {
	// 	err = fmt.Errorf("MatMul tensors' broadcasting failed: %w", err)
	// 	return
	// }

	// r := ct1.matMul(ct2)
	// r.gctx = gradtrack.MatMul(r, ct1, ct2)

	// return r, nil
}

func (t *CUDATensor) Equals(u tensor.Tensor) (are bool, err error) {
	return
	// cu, err := assertCPUTensor(u)
	// if err != nil {
	// 	err = fmt.Errorf("Equals tensors' device validation failed: %w", err)
	// 	return
	// }

	// err = validator.ValidateBinaryFuncDimsMatch(t.dims, cu.dims)
	// if err != nil {
	// 	err = fmt.Errorf("Equals tensors' dimension validation failed: %w", err)
	// 	return
	// }

	// return t.equals(cu), nil
}

func (t *CUDATensor) GradContext() (gctx any) {
	return t.gctx
}

func (t *CUDATensor) ResetGradContext(tracked bool) {
	t.gctx = gradtrack.NewGradContext(tracked)
}

func (t *CUDATensor) Gradient() (g tensor.Tensor) {
	return t.gctx.Gradient()
}
