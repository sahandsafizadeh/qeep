package tensor

/*
Tensor is a data structure with a variable number of dimensions, supporting statistical and linear algebra operations.
As the heart of the qeep project, many design decisions such as the use of build tags, cgo, and device abstraction are driven by this package.

By design:
- Tensor is defined as an interface, allowing for different internal implementations across various devices (e.g., CPU, CUDA).
- Tensor implementations must be immutable.
- Tensor implementations must support automatic gradient computation using the gradtrack package.
- Every tensor must maintain a valid gradient state (i.e., a GradContext), which is the only mutable part of a tensor.
- After backpropagation, the GradContext becomes invalid and must be explicitly reset.
*/
