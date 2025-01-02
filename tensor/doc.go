package tensor

/*
Tensor is a data structure with variable number of dimensions that supports statistics and linear-algebra operations.
By design:
- Tensor is an interface so that it can be implemented on various devices internally.
- Tensor's implementation must be immutable
- Tensor's implementation should support automatic gradient calculation using `gradtrack` package.
- Every Tensor must have a valid gradient state or GradContext; it is the only part of tensors which is not immutable.
- After back-propagation, state of GradContext is not valid and it should be reset.
*/
