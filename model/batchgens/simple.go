package batchgens

import (
	"fmt"
	"math/rand"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Simple struct {
	x         [][]float64
	y         [][]float64
	device    tensor.Device
	shuffle   bool
	batchSize int
	length    int
	index     int
}

type SimpleConfig struct {
	BatchSize int
	Shuffle   bool
	Device    tensor.Device
}

func NewSimple(x [][]float64, y [][]float64, conf *SimpleConfig) (bg *Simple, err error) {
	conf, err = toValidSimpleConfig(conf)
	if err != nil {
		err = fmt.Errorf("Simple config data validation failed: %w", err)
		return
	}

	x, y, err = toValidSimpleData(x, y)
	if err != nil {
		err = fmt.Errorf("Simple input data validation failed: %w", err)
		return
	}

	bg = &Simple{
		x:         x,
		y:         y,
		device:    conf.Device,
		shuffle:   conf.Shuffle,
		batchSize: conf.BatchSize,
		length:    len(x),
	}

	bg.Reset()

	return bg, nil
}

func (bg *Simple) Reset() {
	if bg.shuffle {
		rand.Shuffle(len(bg.x), func(i, j int) {
			bg.x[i], bg.x[j] = bg.x[j], bg.x[i]
			bg.y[i], bg.y[j] = bg.y[j], bg.y[i]
		})
	}

	bg.index = 0
}

func (bg *Simple) Count() (count int) {
	if bg.length%bg.batchSize == 0 {
		return bg.length / bg.batchSize
	} else {
		return bg.length/bg.batchSize + 1
	}
}

func (bg *Simple) HasNext() (ok bool) {
	return bg.index < bg.length
}

func (bg *Simple) NextBatch() (xs []tensor.Tensor, y tensor.Tensor, err error) {
	if !bg.HasNext() {
		err = fmt.Errorf("Simple state validation failed: expected next batch to exist")
		return
	}

	index := bg.index
	length := bg.length
	batchSize := bg.batchSize

	from := index
	to := from + batchSize
	if to > length {
		to = length
	}

	conf := &tensor.Config{Device: bg.device}

	x, err := tensor.TensorOf(bg.x[from:to], conf)
	if err != nil {
		return
	}

	y, err = tensor.TensorOf(bg.y[from:to], conf)
	if err != nil {
		return
	}

	bg.index += bg.batchSize

	return []tensor.Tensor{x}, y, nil
}

/* ----- helpers ----- */

func toValidSimpleConfig(iconf *SimpleConfig) (conf *SimpleConfig, err error) {
	if iconf == nil {
		err = fmt.Errorf("expected config not to be nil")
		return
	}

	conf = new(SimpleConfig)
	*conf = *iconf

	if conf.BatchSize <= 0 {
		err = fmt.Errorf("expected 'BatchSize' to be positive: got (%d)", conf.BatchSize)
		return
	}

	if conf.Device == 0 {
		conf.Device = tensor.CPU
	}

	return conf, nil
}

func toValidSimpleData(ix [][]float64, iy [][]float64) (x [][]float64, y [][]float64, err error) {
	lenx := len(ix)
	leny := len(iy)

	if lenx < 1 || leny < 1 {
		err = fmt.Errorf("expected input slices 'x' and 'y' to have at least one record along dimension (0)")
		return
	}

	if lenx != leny {
		err = fmt.Errorf("expected input slices 'x' and 'y' to have the same number of records along dimension (0): (%d) != (%d)", lenx, leny)
		return
	}

	basex := len(ix[0])
	basey := len(iy[0])

	x = make([][]float64, lenx)
	y = make([][]float64, leny)

	for i := 0; i < lenx; i++ {
		ixi := ix[i]
		iyi := iy[i]

		lenxi := len(ixi)
		lenyi := len(iyi)

		if lenxi < 1 || lenyi < 1 {
			err = fmt.Errorf("expected input slices 'x' and 'y' to have at least one record along dimension (1): got none at position (%d)", i)
			return
		}

		if lenxi != basex {
			err = fmt.Errorf("expected input slice 'x' to have equal length along every record in dimension (1): (%d) != (%d) at position (%d)", lenxi, basex, i)
			return
		}

		if lenyi != basey {
			err = fmt.Errorf("expected input slice 'y' to have equal length along every record in dimension (1): (%d) != (%d) at position (%d)", lenyi, basey, i)
			return
		}

		x[i] = make([]float64, len(ixi))
		y[i] = make([]float64, len(iyi))

		copy(x[i], ixi)
		copy(y[i], iyi)
	}

	return x, y, nil
}
