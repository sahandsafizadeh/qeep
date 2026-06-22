package cputensor

import (
	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
	"gonum.org/v1/gonum/stat/distuv"
)

func newTensorWithElementWiseInit(dims []int, fn elemInitFunc) *CPUTensor {
	t := new(CPUTensor)
	t.dims = make([]int, len(dims))
	copy(t.dims, dims)
	t.strd = util.DimsToStrides(dims)
	t.ofst = make([]int, len(dims))

	t.data = make([]float64, util.DimsToNumElems(dims))
	for i := range t.data {
		t.data[i] = fn()
	}

	return t
}

func constTensor(dims []int, value float64) *CPUTensor {
	return newTensorWithElementWiseInit(dims, func() float64 { return value })
}

func eyeMatrix(n int) *CPUTensor {
	i := 0
	return newTensorWithElementWiseInit([]int{n, n}, func() float64 {
		defer func() { i++ }()

		if i%(n+1) == 0 {
			return 1.
		} else {
			return 0.
		}
	})
}

func uniformRandomTensor(dims []int, l, u float64) *CPUTensor {
	return newTensorWithElementWiseInit(dims, func() float64 { return distuv.Uniform{Min: l, Max: u}.Rand() })
}

func normalRandomTensor(dims []int, u, s float64) *CPUTensor {
	return newTensorWithElementWiseInit(dims, func() float64 { return distuv.Normal{Mu: u, Sigma: s}.Rand() })
}

func tensorFromData(idata any) *CPUTensor {
	var dims []int
	var strd []int
	var ofst []int
	var data []float64

	switch v := idata.(type) {
	case float64:
		dims = []int{}
		strd = util.DimsToStrides(dims)
		ofst = make([]int, len(dims))
		data = make([]float64, util.DimsToNumElems(dims))
		data[0] = v

	case []float64:
		dims = []int{len(v)}
		strd = util.DimsToStrides(dims)
		ofst = make([]int, len(dims))
		data = make([]float64, util.DimsToNumElems(dims))
		for i, v0 := range v {
			data[i*strd[0]] = v0
		}

	case [][]float64:
		dims = []int{len(v), len(v[0])}
		strd = util.DimsToStrides(dims)
		ofst = make([]int, len(dims))
		data = make([]float64, util.DimsToNumElems(dims))
		for i, v0 := range v {
			for j, v1 := range v0 {
				data[i*strd[0]+j*strd[1]] = v1
			}
		}

	case [][][]float64:
		dims = []int{len(v), len(v[0]), len(v[0][0])}
		strd = util.DimsToStrides(dims)
		ofst = make([]int, len(dims))
		data = make([]float64, util.DimsToNumElems(dims))
		for i, v0 := range v {
			for j, v1 := range v0 {
				for k, v2 := range v1 {
					data[i*strd[0]+j*strd[1]+k*strd[2]] = v2
				}
			}
		}

	case [][][][]float64:
		dims = []int{len(v), len(v[0]), len(v[0][0]), len(v[0][0][0])}
		strd = util.DimsToStrides(dims)
		ofst = make([]int, len(dims))
		data = make([]float64, util.DimsToNumElems(dims))
		for i, v0 := range v {
			for j, v1 := range v0 {
				for k, v2 := range v1 {
					for l, v3 := range v2 {
						data[i*strd[0]+j*strd[1]+k*strd[2]+l*strd[3]] = v3
					}
				}
			}
		}

	default:
		panic("invalid input data type: data must have been based on float64 slices")
	}

	return &CPUTensor{
		dims: dims,
		strd: strd,
		ofst: ofst,
		data: data,
	}
}

func tensorFromConcat(ts []*CPUTensor, dim int) *CPUTensor {
	dims := util.ConcatDims(ts, dim)
	dstidx := make([]int, len(dims))
	srcidx := make([]int, len(dims))

	ofsts := make([]int, len(ts)+1)
	for i, t := range ts {
		ofsts[i+1] = ofsts[i] + t.dims[dim]
	}

	return newTensorWithElementWiseInit(dims, func() float64 {
		defer updateElementWiseIndex(dstidx, dims)

		i := 0
		for i < len(ts)-1 && dstidx[dim] >= ofsts[i+1] {
			i++
		}

		copy(srcidx, dstidx)
		srcidx[dim] = dstidx[dim] - ofsts[i]

		return ts[i].at(srcidx)
	})
}

/* ----- helpers ----- */

func updateElementWiseIndex(index []int, dims []int) {
	for i := len(index) - 1; i >= 0; {
		if index[i] < dims[i]-1 {
			index[i]++
			break
		} else {
			index[i] = 0
			i--
		}
	}
}
