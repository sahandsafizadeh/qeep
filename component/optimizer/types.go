package optimizers

import qt "github.com/sahandsafizadeh/qeep/tensor"

type OptimizerFunc func(*qt.Tensor) error
