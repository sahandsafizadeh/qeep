package initializers

import "github.com/sahandsafizadeh/qeep/tensor"

func tensorInitConf(device tensor.Device) *tensor.Config {
	return &tensor.Config{
		Device:    device,
		GradTrack: true,
	}
}
