package forward_test

import (
	"os"
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}

func runTestLogicOnDevices(testLogic func(tinit.Device)) {
	devices := []tinit.Device{
		tinit.CPU,
	}
	for _, dev := range devices {
		testLogic(dev)
	}
}
