package cputensor

import "github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"

type CPUTensor struct {
	ofst int
	strd []int
	dims []int
	data []float64
	gctx *gradtrack.GradContext
}

type reducer interface {
	init()
	feed(index int, value float64)
	result() float64
}

type elemInitFunc func() float64
type scalarUnaryFunc func(float64) float64
type scalarBinaryFunc func(float64, float64) float64
