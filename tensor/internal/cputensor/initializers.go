package cputensor

import (
	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
	"gonum.org/v1/gonum/stat/distuv"
)

func (t *CPUTensor) initWith(initFunc initializerFunc) {

	var fill func([]int, *any)
	fill = func(dims []int, data *any) {
		if len(dims) == 0 {
			*data = initFunc()
			return
		}

		rows := make([]any, dims[0])
		dims = dims[1:]
		for i := range rows {
			fill(dims, &rows[i])
		}

		*data = rows
	}

	fill(t.dims, &t.data)
}

func constTensor(value float64, dims []int) (t *CPUTensor) {
	t = new(CPUTensor)
	t.dims = make([]int, len(dims))
	copy(t.dims, dims)
	t.initWith(func() any { return value })

	return t
}

func eyeMatrix(n int) (t *CPUTensor) {
	t = new(CPUTensor)
	t.dims = []int{n, n}
	t.initWith(eyeElemGenerator(n))

	return t
}

func uniformRandomTensor(l, u float64, dims []int) (t *CPUTensor) {
	t = new(CPUTensor)
	t.dims = make([]int, len(dims))
	copy(t.dims, dims)
	t.initWith(func() any {
		return distuv.Uniform{Min: l, Max: u}.Rand()
	})

	return t
}

func normalRandomTensor(u, s float64, dims []int) (t *CPUTensor) {
	t = new(CPUTensor)
	t.dims = make([]int, len(dims))
	copy(t.dims, dims)
	t.initWith(func() any {
		return distuv.Normal{Mu: u, Sigma: s}.Rand()
	})

	return t
}

func initTensorFromData(data any) (t *CPUTensor) {
	var dims []int
	var tensorData any

	switch v := data.(type) {
	case float64:
		dims = []int{}
		tensorData = v

	case []float64:
		d0 := len(v)
		dims = []int{d0}
		data0 := make([]any, d0)
		for i, v0 := range v {
			data0[i] = v0
		}
		tensorData = data0

	case [][]float64:
		d0 := len(v)
		d1 := len(v[0])
		dims = []int{d0, d1}
		data0 := make([]any, d0)
		for i, v0 := range v {
			data1 := make([]any, d1)
			for i, v1 := range v0 {
				data1[i] = v1
			}
			data0[i] = data1
		}
		tensorData = data0

	case [][][]float64:
		d0 := len(v)
		d1 := len(v[0])
		d2 := len(v[0][0])
		dims = []int{d0, d1, d2}
		data0 := make([]any, d0)
		for i, v0 := range v {
			data1 := make([]any, d1)
			for i, v1 := range v0 {
				data2 := make([]any, d2)
				for i, v2 := range v1 {
					data2[i] = v2
				}
				data1[i] = data2
			}
			data0[i] = data1
		}
		tensorData = data0

	case [][][][]float64:
		d0 := len(v)
		d1 := len(v[0])
		d2 := len(v[0][0])
		d3 := len(v[0][0][0])
		dims = []int{d0, d1, d2, d3}
		data0 := make([]any, d0)
		for i, v0 := range v {
			data1 := make([]any, d1)
			for i, v1 := range v0 {
				data2 := make([]any, d2)
				for i, v2 := range v1 {
					data3 := make([]any, d3)
					for i, v3 := range v2 {
						data3[i] = v3
					}
					data2[i] = data3
				}
				data1[i] = data2
			}
			data0[i] = data1
		}
		tensorData = data0

	default:
		panic("invalid input data type: data must have been based on float64 slices")
	}

	return &CPUTensor{
		data: tensorData,
		dims: dims,
	}
}

func initConcatResultTensor(ts []*CPUTensor, dim int) (o *CPUTensor) {
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

	o = new(CPUTensor)
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
