package cputensor

import (
	"math"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func (t *CPUTensor) sum() (value float64) {
	return t.reduceByAssociativeFunc(func(a, b float64) float64 { return a + b }, 0.)
}

func (t *CPUTensor) max() (value float64) {
	return t.reduceByAssociativeFunc(func(a, b float64) float64 {
		if a > b {
			return a
		} else {
			return b
		}
	}, math.Inf(-1))
}

func (t *CPUTensor) min() (value float64) {
	return t.reduceByAssociativeFunc(func(a, b float64) float64 {
		if a < b {
			return a
		} else {
			return b
		}
	}, math.Inf(+1))
}

func (t *CPUTensor) avg() (value float64) {
	return t.sum() / float64(t.numElems())
}

func (t *CPUTensor) _var() (value float64) {
	xBar := t.mean()
	sigma := t.reduceByAssociativeFunc(func(s, x float64) float64 { return s + math.Pow(x-xBar, 2) }, 0.)

	n := float64(t.numElems())
	if n > 1 {
		return sigma / (n - 1)
	} else {
		return 0.
	}
}

func (t *CPUTensor) std() (value float64) {
	return math.Sqrt(t._var())
}

func (t *CPUTensor) mean() (value float64) {
	return t.avg()
}

func (t *CPUTensor) sumAlong(dim int) (o *CPUTensor) {
	return t.reduceDimUsingFunc(dim, func(u *CPUTensor) float64 { return u.sum() })
}

func (t *CPUTensor) maxAlong(dim int) (o *CPUTensor) {
	return t.reduceDimUsingFunc(dim, func(u *CPUTensor) float64 { return u.max() })
}

func (t *CPUTensor) minAlong(dim int) (o *CPUTensor) {
	return t.reduceDimUsingFunc(dim, func(u *CPUTensor) float64 { return u.min() })
}

func (t *CPUTensor) avgAlong(dim int) (o *CPUTensor) {
	return t.reduceDimUsingFunc(dim, func(u *CPUTensor) float64 { return u.avg() })
}

func (t *CPUTensor) varAlong(dim int) (o *CPUTensor) {
	return t.reduceDimUsingFunc(dim, func(u *CPUTensor) float64 { return u._var() })
}

func (t *CPUTensor) stdAlong(dim int) (o *CPUTensor) {
	return t.reduceDimUsingFunc(dim, func(u *CPUTensor) float64 { return u.std() })
}

func (t *CPUTensor) meanAlong(dim int) (o *CPUTensor) {
	return t.reduceDimUsingFunc(dim, func(u *CPUTensor) float64 { return u.mean() })
}

/* ----- helpers ----- */

func (t *CPUTensor) reduceByAssociativeFunc(af scalarBinaryFunc, identity float64) (value float64) {
	value = identity

	var trav func([]int, any)
	trav = func(dims []int, data any) {
		if len(dims) == 0 {
			value = af(value, data.(float64))
			return
		}

		dims = dims[1:]
		rows := data.([]any)
		for i := range rows {
			trav(dims, rows[i])
		}
	}

	trav(t.dims, t.data)
	return value
}

func (t *CPUTensor) reduceDimUsingFunc(dim int, trf tensorReducerFunc) (o *CPUTensor) {
	dims := squeezeDims(dim, t.dims)
	elemGen := t.linearElemGeneratorWithReducedDim(dim, trf)

	o = new(CPUTensor)
	o.dims = dims
	o.initWith(elemGen)

	return o
}

func (t *CPUTensor) linearElemGeneratorWithReducedDim(dim int, trf tensorReducerFunc) initializerFunc {
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

		return trf(row)
	}
}
