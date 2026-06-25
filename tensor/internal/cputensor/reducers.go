package cputensor

import (
	"math"

	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

func (t *CPUTensor) sum() float64 {
	return t.reduceByAssociativeFunc(func(a, b reducerPair) reducerPair {
		return reducerPair{value: a.value + b.value}
	}, unwrapValue, 0.)
}

func (t *CPUTensor) max() float64 {
	return t.reduceByAssociativeFunc(func(a, b reducerPair) reducerPair {
		if a.value >= b.value {
			return a
		} else {
			return b
		}
	}, unwrapValue, math.Inf(-1))
}

func (t *CPUTensor) min() float64 {
	return t.reduceByAssociativeFunc(func(a, b reducerPair) reducerPair {
		if a.value <= b.value {
			return a
		} else {
			return b
		}
	}, unwrapValue, math.Inf(+1))
}

func (t *CPUTensor) avg() float64 {
	n := float64(t.numElems())
	return t.sum() / n
}

func (t *CPUTensor) _var() float64 {
	n := float64(t.numElems())
	if n <= 1 {
		return 0.
	}

	xBar := t.mean()
	return t.reduceByAssociativeFunc(func(s, x reducerPair) reducerPair {
		xdiff := x.value - xBar
		xdiff2 := xdiff * xdiff
		return reducerPair{value: s.value + (xdiff2 / (n - 1))}
	}, unwrapValue, 0.)
}

func (t *CPUTensor) std() float64 {
	return math.Sqrt(t._var())
}

func (t *CPUTensor) mean() float64 {
	return t.avg()
}

func (t *CPUTensor) argmax(dim int) *CPUTensor {
	return t.reduceDimByAssociativeFunc(dim, func(a, b reducerPair) reducerPair {
		if a.value >= b.value {
			return a
		} else {
			return b
		}
	}, unwrapIndex, math.Inf(-1))
}

func (t *CPUTensor) argmin(dim int) *CPUTensor {
	return t.reduceDimByAssociativeFunc(dim, func(a, b reducerPair) reducerPair {
		if a.value <= b.value {
			return a
		} else {
			return b
		}
	}, unwrapIndex, math.Inf(+1))
}

func (t *CPUTensor) sumAlong(dim int) *CPUTensor {
	return t.reduceDimByAssociativeFunc(dim, func(a, b reducerPair) reducerPair {
		return reducerPair{value: a.value + b.value}
	}, unwrapValue, 0.)
}

func (t *CPUTensor) maxAlong(dim int) *CPUTensor {
	return t.reduceDimByAssociativeFunc(dim, func(a, b reducerPair) reducerPair {
		if a.value >= b.value {
			return a
		} else {
			return b
		}
	}, unwrapValue, math.Inf(-1))
}

func (t *CPUTensor) minAlong(dim int) *CPUTensor {
	return t.reduceDimByAssociativeFunc(dim, func(a, b reducerPair) reducerPair {
		if a.value <= b.value {
			return a
		} else {
			return b
		}
	}, unwrapValue, math.Inf(+1))
}

func (t *CPUTensor) avgAlong(dim int) *CPUTensor {
	return t.reduceDimUsingTensorFunc(dim, func(u *CPUTensor) float64 { return u.avg() })
}

func (t *CPUTensor) varAlong(dim int) *CPUTensor {
	return t.reduceDimUsingTensorFunc(dim, func(u *CPUTensor) float64 { return u._var() })
}

func (t *CPUTensor) stdAlong(dim int) *CPUTensor {
	return t.reduceDimUsingTensorFunc(dim, func(u *CPUTensor) float64 { return u.std() })
}

func (t *CPUTensor) meanAlong(dim int) *CPUTensor {
	return t.avgAlong(dim)
}

/* ----- helpers ----- */

func (t *CPUTensor) reduceByAssociativeFunc(af reducerFunc, uf reducerUnwrapFunc, identityValue float64) float64 {
	index := make([]int, len(t.dims))

	result := reducerPair{0, identityValue}
	for i := range t.numElems() {
		result = af(result, reducerPair{i, t.at(index)})
		updateElementWiseIndex(index, t.dims)
	}

	return uf(result)
}

func (t *CPUTensor) reduceDimByAssociativeFunc(dim int, af reducerFunc, uf reducerUnwrapFunc, identity float64) *CPUTensor {
	dims := util.SqueezeDims(dim, t.dims)
	dstidx := make([]int, len(dims))
	srcidx := make([]int, len(t.dims))

	return newTensorWithElementWiseInit(dims, func() float64 {
		defer updateElementWiseIndex(dstidx, dims)

		i := 0
		for j := range srcidx {
			if j != dim {
				srcidx[j] = dstidx[i]
				i++
			}
		}

		result := reducerPair{0, identity}
		for i := range t.dims[dim] {
			srcidx[dim] = i
			result = af(result, reducerPair{i, t.at(srcidx)})
		}

		return uf(result)
	})
}

/* ----- unwrappers ----- */

func unwrapIndex(r reducerPair) float64 {
	return float64(r.index)
}

func unwrapValue(r reducerPair) float64 {
	return r.value
}
