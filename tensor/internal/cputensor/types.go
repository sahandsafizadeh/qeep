package cputensor

import "github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"

type CPUTensor struct {
	data any
	dims []int32
	gctx *gradtrack.GradContext
}

type initializerFunc func() any
type scalarUnaryFunc func(float64) float64
type scalarBinaryFunc func(float64, float64) float64
type tensorReducerFunc func(*CPUTensor) float64
