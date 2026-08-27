package cputensor

import (
	"math"

	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

func (t *CPUTensor) sum() float64 {
	return t.reduceAll(new(sumReducer))
}

func (t *CPUTensor) max() float64 {
	return t.reduceAll(new(maxReducer))
}

func (t *CPUTensor) min() float64 {
	return t.reduceAll(new(minReducer))
}

func (t *CPUTensor) avg() float64 {
	return t.reduceAll(new(avgReducer))
}

func (t *CPUTensor) _var() float64 {
	return t.reduceAll(new(varReducer))
}

func (t *CPUTensor) std() float64 {
	return t.reduceAll(new(stdReducer))
}

func (t *CPUTensor) mean() float64 {
	return t.avg()
}

func (t *CPUTensor) argmax(dim int) *CPUTensor {
	return t.reduceDim(dim, new(argmaxReducer))
}

func (t *CPUTensor) argmin(dim int) *CPUTensor {
	return t.reduceDim(dim, new(argminReducer))
}

func (t *CPUTensor) sumAlong(dim int) *CPUTensor {
	return t.reduceDim(dim, new(sumReducer))
}

func (t *CPUTensor) maxAlong(dim int) *CPUTensor {
	return t.reduceDim(dim, new(maxReducer))
}

func (t *CPUTensor) minAlong(dim int) *CPUTensor {
	return t.reduceDim(dim, new(minReducer))
}

func (t *CPUTensor) avgAlong(dim int) *CPUTensor {
	return t.reduceDim(dim, new(avgReducer))
}

func (t *CPUTensor) varAlong(dim int) *CPUTensor {
	return t.reduceDim(dim, new(varReducer))
}

func (t *CPUTensor) stdAlong(dim int) *CPUTensor {
	return t.reduceDim(dim, new(stdReducer))
}

func (t *CPUTensor) meanAlong(dim int) *CPUTensor {
	return t.avgAlong(dim)
}

/* ----- helpers ----- */

func (t *CPUTensor) reduceAll(r reducer) float64 {
	index := make([]int, len(t.dims))

	r.init()
	for i := range t.numElems() {
		r.feed(i, t.at(index))
		updateElementWiseIndex(index, t.dims)
	}

	return r.result()
}

func (t *CPUTensor) reduceDim(dim int, r reducer) *CPUTensor {
	dims := util.SqueezeDims(dim, t.dims)
	dstidx := make([]int, len(dims))
	srcidx := make([]int, len(t.dims))

	return newTensorWithElementWiseInit(dims, func() float64 {
		defer updateElementWiseIndex(dstidx, dims)

		i := 0
		for j := range srcidx {
			if j == dim {
				continue
			}

			srcidx[j] = dstidx[i]
			i++
		}

		r.init()
		for i := range t.dims[dim] {
			srcidx[dim] = i
			r.feed(i, t.at(srcidx))
		}

		return r.result()
	})
}

/* ---------- reducer implementations ---------- */

// ----- sum -----
type sumReducer struct {
	value float64
}

func (r *sumReducer) init() {
	r.value = 0
}

func (r *sumReducer) feed(_ int, value float64) {
	r.value += value
}

func (r *sumReducer) result() float64 {
	return r.value
}

// ----- max -----
type maxReducer struct {
	value float64
}

func (r *maxReducer) init() {
	r.value = math.Inf(-1)
}

func (r *maxReducer) feed(_ int, value float64) {
	if value > r.value {
		r.value = value
	}
}

func (r *maxReducer) result() float64 {
	return r.value
}

// ----- min -----
type minReducer struct {
	value float64
}

func (r *minReducer) init() {
	r.value = math.Inf(+1)
}

func (r *minReducer) feed(_ int, value float64) {
	if value < r.value {
		r.value = value
	}
}

func (r *minReducer) result() float64 {
	return r.value
}

// ----- argmax -----
type argmaxReducer struct {
	index int
	value float64
}

func (r *argmaxReducer) init() {
	r.index = -1
	r.value = math.Inf(-1)
}

func (r *argmaxReducer) feed(index int, value float64) {
	if value > r.value {
		r.index = index
		r.value = value
	}
}

func (r *argmaxReducer) result() float64 {
	return float64(r.index)
}

// ----- argmin -----
type argminReducer struct {
	index int
	value float64
}

func (r *argminReducer) init() {
	r.index = -1
	r.value = math.Inf(+1)
}

func (r *argminReducer) feed(index int, value float64) {
	if value < r.value {
		r.index = index
		r.value = value
	}
}

func (r *argminReducer) result() float64 {
	return float64(r.index)
}

// ----- avg (Welford) -----
type avgReducer struct {
	count int
	mean  float64
}

func (r *avgReducer) init() {
	r.count = 0
	r.mean = 0.
}

func (r *avgReducer) feed(_ int, value float64) {
	r.count++
	delta := value - r.mean
	r.mean += delta / float64(r.count)
}

func (r *avgReducer) result() float64 {
	return r.mean
}

// ----- var (Welford) -----
type varReducer struct {
	count int
	mean  float64
	m2    float64
}

func (r *varReducer) init() {
	r.count = 0
	r.mean = 0
	r.m2 = 0
}

func (r *varReducer) feed(_ int, value float64) {
	r.count++
	delta := value - r.mean
	r.mean += delta / float64(r.count)
	delta2 := value - r.mean
	r.m2 += delta * delta2
}

func (r *varReducer) result() float64 {
	if r.count <= 1 {
		return 0.
	}

	return r.m2 / float64(r.count-1)
}

// ----- std (Welford) -----
type stdReducer struct {
	varReducer
}

func (r *stdReducer) result() float64 {
	return math.Sqrt(r.varReducer.result())
}
