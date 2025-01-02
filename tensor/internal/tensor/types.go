package tensor

/*
Tensor is a data structure with variable number of dimensions that supports statistics and linear-algebra operations.
By design:
- Tensor is an interface so that it can be implemented on various devices internally.
- Tensor's implementation must be immutable
- Tensor's implementation must support automatic gradient calculation using `gradtrack` package.
- Every Tensor must have a valid gradient state or `GradContext`; it is the only part of tensors which is not immutable.
- After back-propagation, state of GradContext is not valid and it must be reset.
*/

type Tensor interface {
	/*--------------- accessors ---------------*/
	NElems() int
	Shape() []int
	At(index ...int) (float64, error)
	Slice(index []Range) (Tensor, error)
	Patch(index []Range, source Tensor) (Tensor, error)

	/*------------ shape modifiers ------------*/
	Transpose() (Tensor, error)
	Reshape(shape []int) (Tensor, error)
	UnSqueeze(dim int) (Tensor, error)
	Squeeze(dim int) (Tensor, error)
	Flatten(fromDim int) (Tensor, error)
	Broadcast(shape []int) (Tensor, error)

	/*--------------- reducers ----------------*/
	Sum() float64
	Max() float64
	Min() float64
	Avg() float64
	Var() float64
	Std() float64
	Mean() float64
	Argmax(dim int) (Tensor, error)
	Argmin(dim int) (Tensor, error)
	SumAlong(dim int) (Tensor, error)
	MaxAlong(dim int) (Tensor, error)
	MinAlong(dim int) (Tensor, error)
	AvgAlong(dim int) (Tensor, error)
	VarAlong(dim int) (Tensor, error)
	StdAlong(dim int) (Tensor, error)
	MeanAlong(dim int) (Tensor, error)

	/*--------------- operators ---------------*/
	Scale(float64) Tensor
	Pow(float64) Tensor
	Exp() Tensor
	Log() Tensor
	Sin() Tensor
	Cos() Tensor
	Tan() Tensor
	Sinh() Tensor
	Cosh() Tensor
	Tanh() Tensor
	Eq(Tensor) (Tensor, error)
	Ne(Tensor) (Tensor, error)
	Gt(Tensor) (Tensor, error)
	Ge(Tensor) (Tensor, error)
	Lt(Tensor) (Tensor, error)
	Le(Tensor) (Tensor, error)
	ElMax(Tensor) (Tensor, error)
	ElMin(Tensor) (Tensor, error)
	Add(Tensor) (Tensor, error)
	Sub(Tensor) (Tensor, error)
	Mul(Tensor) (Tensor, error)
	Div(Tensor) (Tensor, error)
	Dot(Tensor) (Tensor, error)
	MatMul(Tensor) (Tensor, error)
	Equals(Tensor) (bool, error)

	/*--------------- gradients ---------------*/
	GradContext() any
	ResetGradContext(bool)
	Gradient() Tensor
}

type Range struct {
	From int
	To   int
}
