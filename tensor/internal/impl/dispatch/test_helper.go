package dispatch

import (
	"github.com/sahandsafizadeh/qeep/tensor/internal/core"
	"github.com/sahandsafizadeh/qeep/tensor/internal/impl/cudatensor"
)

func RunTestLogicOnDevices(testLogic func(core.Device)) {
	devices := []core.Device{core.CPU}

	if cudatensor.IsAvailable {
		devices = append(devices, core.CUDA)
	}

	for _, dev := range devices {
		testLogic(dev)
	}
}
