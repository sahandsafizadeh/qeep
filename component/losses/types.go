package losses

import qt "github.com/sahandsafizadeh/qeep/tensor"

type LossFunc func(yp qt.Tensor, yt qt.Tensor) (qt.Tensor, error)
