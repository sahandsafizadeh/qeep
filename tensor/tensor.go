package tensor

type Tensor interface {
	/*--------------- accessors ---------------*/
	NElems() int64
	Shape() []int32
	At(index ...int32) (float64, error)
	Slice(index []Range) (Tensor, error)
	Patch(index []Range, source Tensor) (Tensor, error)

	/*------------ shape modifiers ------------*/
	Transpose() (Tensor, error)
	Reshape(shape ...int32) (Tensor, error)
	UnSqueeze(dim int32) (Tensor, error)
	Squeeze(dim int32) (Tensor, error)
	Flatten(fromDim int32) (Tensor, error)
	Broadcast(shape ...int32) (Tensor, error)

	/*--------------- reducers ----------------*/
	Sum() float64
	Max() float64
	Min() float64
	Avg() float64
	Var() float64
	Std() float64
	Mean() float64
	SumAlong(dim int32) (Tensor, error)
	MaxAlong(dim int32) (Tensor, error)
	MinAlong(dim int32) (Tensor, error)
	AvgAlong(dim int32) (Tensor, error)
	VarAlong(dim int32) (Tensor, error)
	StdAlong(dim int32) (Tensor, error)
	MeanAlong(dim int32) (Tensor, error)

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
	ElMin(Tensor) (Tensor, error)
	ElMax(Tensor) (Tensor, error)
	Add(Tensor) (Tensor, error)
	Sub(Tensor) (Tensor, error)
	Mul(Tensor) (Tensor, error)
	Div(Tensor) (Tensor, error)
	Dot(Tensor) (Tensor, error)
	MatMul(Tensor) (Tensor, error)
	Equals(Tensor) (bool, error)

	/*--------------- gradients ---------------*/
	GradContext() any
	Gradient() Tensor
	Detach() Tensor
}

type Range struct {
	From int32
	To   int32
}
