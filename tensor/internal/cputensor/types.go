package cputensor

import "github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"

type CPUTensor struct {
	dims []int
	strd []int
	data []float64
	gctx *gradtrack.GradContext
}

type reducerPair struct {
	index int
	value float64
}

type elemInitFunc func(i int) float64
type scalarUnaryFunc func(float64) float64
type scalarBinaryFunc func(float64, float64) float64
type reducerFunc func(reducerPair, reducerPair) reducerPair
type reducerUnwrapFunc func(reducerPair) float64
type reducerTensorFunc func(*CPUTensor) float64
