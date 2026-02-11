package layers

import "github.com/sahandsafizadeh/qeep/tensor"

// Initializer creates a tensor of the given shape on the given device (e.g. for layer weights).
type Initializer interface {
	Init(shape []int, device tensor.Device) (tensor.Tensor, error)
}

// Weight holds a pointer to a trainable tensor and whether it is updated by the optimizer.
type Weight struct {
	Value     *tensor.Tensor
	Trainable bool
}
