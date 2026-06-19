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

	t.data = make([]float64, util.DimsToNumElems(dims))
	for i := range t.data {
		t.data[i] = fn(i)
	}

	return t
}

func constTensor(dims []int, value float64) *CPUTensor {
	return newTensorWithElementWiseInit(dims, func(int) float64 {
		return value
	})
}

func eyeMatrix(n int) *CPUTensor {
	return newTensorWithElementWiseInit([]int{n, n}, func(i int) float64 {
		if i%(n+1) == 0 {
			return 1.
		} else {
			return 0.
		}
	})
}

func uniformRandomTensor(dims []int, l, u float64) *CPUTensor {
	return newTensorWithElementWiseInit(dims, func(int) float64 {
		return distuv.Uniform{Min: l, Max: u}.Rand()
	})
}

func normalRandomTensor(dims []int, u, s float64) *CPUTensor {
	return newTensorWithElementWiseInit(dims, func(int) float64 {
		return distuv.Normal{Mu: u, Sigma: s}.Rand()
	})
}

func tensorFromData(idata any) *CPUTensor {
	var dims []int
	var strd []int
	var data []float64

	switch v := idata.(type) {
	case float64:
		dims = []int{}
		strd = util.DimsToStrides(dims)
		data = make([]float64, util.DimsToNumElems(dims))
		data[0] = v

	case []float64:
		dims = []int{len(v)}
		strd = util.DimsToStrides(dims)
		data = make([]float64, util.DimsToNumElems(dims))
		for i, v0 := range v {
			data[i*strd[0]] = v0
		}

	case [][]float64:
		dims = []int{len(v), len(v[0])}
		strd = util.DimsToStrides(dims)
		data = make([]float64, util.DimsToNumElems(dims))
		for i, v0 := range v {
			for j, v1 := range v0 {
				data[i*strd[0]+j*strd[1]] = v1
			}
		}

	case [][][]float64:
		dims = []int{len(v), len(v[0]), len(v[0][0])}
		strd = util.DimsToStrides(dims)
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
		data: data,
	}
}

func tensorFromConcat(ts []*CPUTensor, dim int) *CPUTensor {
	tsDataCopy := make([]any, len(ts))
	for i, t := range ts {
		tc := t.slice(nil)
		tsDataCopy[i] = tc.data
	}

	var fillCat func([]int, *any, []any, int)
	fillCat = func(dims []int, data *any, seeds []any, depth int) {
		if depth == dim {
			catData := make([]any, 0, dims[0])
			for _, seed := range seeds {
				catData = append(catData, seed.([]any)...)
			}

			*data = catData
			return
		}

		rows := make([]any, dims[0])
		dims = dims[1:]
		depth++

		for i := range rows {
			seedRows := make([]any, 0, len(rows))
			for _, seed := range seeds {
				seedRows = append(seedRows, seed.([]any)[i])
			}
			fillCat(dims, &rows[i], seedRows, depth)
		}

		*data = rows
	}

	o := new(CPUTensor)
	o.dims = util.ConcatDims(ts, dim)
	fillCat(o.dims, &o.data, tsDataCopy, 0)

	return o
}

/* ----- helpers ----- */

func eyeElemGenerator(n int) initializerFunc {
	var state int

	return func() any {
		atDiag := state%(n+1) == 0
		state++

		if atDiag {
			return 1.
		} else {
			return 0.
		}
	}
}
