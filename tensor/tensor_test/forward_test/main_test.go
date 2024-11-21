package forward_test

import (
	"os"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}

func runTestLogicOnDevices(testLogic func(tensor.Device)) {
	devices := []tensor.Device{
		tensor.CPU,
	}
	for _, dev := range devices {
		testLogic(dev)
	}
}
