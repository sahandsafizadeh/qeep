package cputensor

import (
	"math"

	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

func (t *CPUTensor) sum() (data float64) {
	return t.reduceByAssociativeFunc(func(a, b reducerPair) reducerPair {
		return reducerPair{value: a.value + b.value}
	}, unwrapValue, 0.)
}

func (t *CPUTensor) max(uf reducerUnwrapFunc) (data float64) {
	return t.reduceByAssociativeFunc(func(a, b reducerPair) reducerPair {
		if a.value >= b.value {
			return a
		} else {
			return b
		}
	}, uf, math.Inf(-1))
}

func (t *CPUTensor) min(uf reducerUnwrapFunc) (data float64) {
	return t.reduceByAssociativeFunc(func(a, b reducerPair) reducerPair {
		if a.value <= b.value {
			return a
		} else {
			return b
		}
	}, uf, math.Inf(+1))
}

func (t *CPUTensor) avg() (data float64) {
	return t.sum() / float64(t.numElems())
}

func (t *CPUTensor) _var() (data float64) {
	xBar := t.mean()
	sigma := t.reduceByAssociativeFunc(func(s, x reducerPair) reducerPair {
		xdiff2 := math.Pow(x.value-xBar, 2)
		return reducerPair{value: s.value + xdiff2}
	}, unwrapValue, 0.)

	n := float64(t.numElems())
	if n > 1 {
		return sigma / (n - 1)
	} else {
		return 0.
	}
}

func (t *CPUTensor) std() (data float64) {
	return math.Sqrt(t._var())
}

func (t *CPUTensor) mean() (data float64) {
	return t.avg()
}

func (t *CPUTensor) argmax(dim int) (o *CPUTensor) {
	return t.reduceDimUsingTensorFunc(dim, func(u *CPUTensor) float64 { return u.max(unwrapIndex) })
}

func (t *CPUTensor) argmin(dim int) (o *CPUTensor) {
	return t.reduceDimUsingTensorFunc(dim, func(u *CPUTensor) float64 { return u.min(unwrapIndex) })
}

func (t *CPUTensor) sumAlong(dim int) (o *CPUTensor) {
	return t.reduceDimUsingTensorFunc(dim, func(u *CPUTensor) float64 { return u.sum() })
}

func (t *CPUTensor) maxAlong(dim int) (o *CPUTensor) {
	return t.reduceDimUsingTensorFunc(dim, func(u *CPUTensor) float64 { return u.max(unwrapValue) })
}

func (t *CPUTensor) minAlong(dim int) (o *CPUTensor) {
	return t.reduceDimUsingTensorFunc(dim, func(u *CPUTensor) float64 { return u.min(unwrapValue) })
}

func (t *CPUTensor) avgAlong(dim int) (o *CPUTensor) {
	return t.reduceDimUsingTensorFunc(dim, func(u *CPUTensor) float64 { return u.avg() })
}

func (t *CPUTensor) varAlong(dim int) (o *CPUTensor) {
	return t.reduceDimUsingTensorFunc(dim, func(u *CPUTensor) float64 { return u._var() })
}

func (t *CPUTensor) stdAlong(dim int) (o *CPUTensor) {
	return t.reduceDimUsingTensorFunc(dim, func(u *CPUTensor) float64 { return u.std() })
}

func (t *CPUTensor) meanAlong(dim int) (o *CPUTensor) {
	return t.reduceDimUsingTensorFunc(dim, func(u *CPUTensor) float64 { return u.mean() })
}

/* ----- helpers ----- */

func (t *CPUTensor) reduceByAssociativeFunc(af reducerFunc, uf reducerUnwrapFunc, identityValue float64) (reducedData float64) {
	result := reducerPair{value: identityValue}
	index := 0

	var trav func([]int, any)
	trav = func(dims []int, data any) {
		if len(dims) == 0 {
			result = af(result, reducerPair{
				index: index,
				value: data.(float64),
			})
			index++
			return
		}

		dims = dims[1:]
		rows := data.([]any)
		for i := range rows {
			trav(dims, rows[i])
		}
	}

	trav(t.dims, t.data)

	return uf(result)
}

func (t *CPUTensor) reduceDimUsingTensorFunc(dim int, rtf reducerTensorFunc) (o *CPUTensor) {
	dims := util.SqueezeDims(dim, t.dims)
	elemGen := t.linearElemGeneratorWithReducedDim(dim, rtf)

	o = new(CPUTensor)
	o.dims = dims
	o.initWith(elemGen)

	return o
}

func (t *CPUTensor) linearElemGeneratorWithReducedDim(dim int, rtf reducerTensorFunc) initializerFunc {
	state := make([]tensor.Range, len(t.dims))
	for i := 0; i < len(state); i++ {
		state[i].From = 0
		state[i].To = 1
	}
	state[dim].To = t.dims[dim]

	return func() any {
		row := t.slice(state)

		i := len(t.dims) - 1
		for i >= 0 {
			if i == dim {
				i--
				continue
			}

			if state[i].To < t.dims[i] {
				state[i].From++
				state[i].To++
				break
			} else {
				state[i].From = 0
				state[i].To = 1
				i--
			}
		}

		return rtf(row)
	}
}

/* ----- unwrappers ----- */

func unwrapIndex(r reducerPair) float64 {
	return float64(r.index)
}

func unwrapValue(r reducerPair) float64 {
	return r.value
}
