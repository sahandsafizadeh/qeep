package cudatensor_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func BenchmarkMemory(b *testing.B) {
	for b.Loop() {
		ten, err := tensor.RandN([]int{256, 128, 3}, 0., 0.5, &tensor.Config{
			Device:    tensor.CUDA,
			GradTrack: false,
		})
		if err != nil {
			b.Fatal(err)
		}

		_, err = ten.At(128, 32, 1)
		if err != nil {
			b.Fatal(err)
		}
	}
}

/*
go test -bench=. -benchtime=60s -benchmem
1. always:			BenchmarkMemory-8         137240            542985 ns/op
2. first+0.9+2ice:	BenchmarkMemory-8         132416            548662 ns/op
*/
