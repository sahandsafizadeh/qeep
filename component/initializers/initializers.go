package initializers

import "github.com/sahandsafizadeh/qeep/tensor/tinit"

func tensorInitConf() *tinit.Config {
	return &tinit.Config{
		Device:    tinit.CPU,
		GradTrack: true,
	}
}
