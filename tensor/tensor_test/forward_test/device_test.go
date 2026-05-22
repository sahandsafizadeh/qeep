package forward_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestDevice(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("Full(nil, 0) scalar tensor / Device() / returns the device it was created on", func(t *testing.T) {
			ten, err := tensor.Full(nil, 0., &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}

			if d := ten.Device(); d != dev {
				t.Fatalf("expected tensor's device to be (%s), got (%s)", dev, d)
			}
		})
	})
}
