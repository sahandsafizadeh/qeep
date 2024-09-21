package initializers

import "github.com/sahandsafizadeh/qeep/tensor/tinit"

func tensorInitConf(dev tinit.Device) *tinit.Config {
	return &tinit.Config{
		Device:    dev,
		GradTrack: true,
	}
}
