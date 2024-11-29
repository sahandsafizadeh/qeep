package initializers

import "github.com/sahandsafizadeh/qeep/tensor"

func tensorInitConf() *tensor.Config {
	return &tensor.Config{
		Device:    tensor.CPU,
		GradTrack: true,
	}
}
