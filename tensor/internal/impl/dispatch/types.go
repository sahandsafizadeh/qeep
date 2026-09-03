package dispatch

import "github.com/sahandsafizadeh/qeep/tensor/internal/core"

type TestHelperAllDeviceFunc func(core.Device)
type TestHelperCrossDeviceFunc func(core.Device, core.Device)
