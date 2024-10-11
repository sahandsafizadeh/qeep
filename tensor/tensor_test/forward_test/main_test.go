package forward_test

import (
	"os"
	"testing"

	qti "github.com/sahandsafizadeh/qeep/tensor/tinit"
)

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}

func runTestLogicOnDevices(testLogic func(qti.Device)) {
	devices := []qti.Device{
		qti.CPU,
	}
	for _, dev := range devices {
		testLogic(dev)
	}
}
