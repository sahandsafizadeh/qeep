package batchgens

import (
	"fmt"
	"math/rand/v2"

	"github.com/sahandsafizadeh/qeep/tensor"
)

// Simple is a batch generator that yields (x, y) batches from in-memory slices [][]float64.
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
		return bg, fmt.Errorf("Simple config data validation failed: %w", err)
	}

	x, y, err = toValidSimpleData(x, y)
	if err != nil {
		return bg, fmt.Errorf("Simple input data validation failed: %w", err)
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

func (bg *Simple) Count() int {
	if bg.length%bg.batchSize == 0 {
		return bg.length / bg.batchSize
	} else {
		return bg.length/bg.batchSize + 1
	}
}

func (bg *Simple) HasNext() bool {
	return bg.index < bg.length
}

func (bg *Simple) NextBatch() (xs []tensor.Tensor, y tensor.Tensor, err error) {
	if !bg.HasNext() {
		return xs, y, fmt.Errorf("Simple state validation failed: expected next batch to exist")
	}

	xs, y, err = bg.nextBatch()
	if err != nil {
		return xs, y, fmt.Errorf("Simple batch fetching failed: %w", err)
	}

	return xs, y, nil
}

func (bg *Simple) nextBatch() (xs []tensor.Tensor, y tensor.Tensor, err error) {
	index := bg.index
	length := bg.length
	batchSize := bg.batchSize

	from := index
	to := min(from+batchSize, length)

	conf := &tensor.Config{Device: bg.device}

	x, err := tensor.Of(bg.x[from:to], conf)
	if err != nil {
		return xs, y, err
	}

	y, err = tensor.Of(bg.y[from:to], conf)
	if err != nil {
		return xs, y, err
	}

	bg.index += bg.batchSize

	return []tensor.Tensor{x}, y, nil
}

/* ----- helpers ----- */

func toValidSimpleConfig(iconf *SimpleConfig) (conf *SimpleConfig, err error) {
	if iconf == nil {
		return conf, fmt.Errorf("expected config not to be nil")
	}

	conf = new(SimpleConfig)
	*conf = *iconf

	if conf.Device == 0 {
		conf.Device = tensor.CPU
	}

	if conf.BatchSize <= 0 {
		return conf, fmt.Errorf("expected 'BatchSize' to be positive: got (%d)", conf.BatchSize)
	}

	return conf, nil
}

func toValidSimpleData(ix [][]float64, iy [][]float64) (x [][]float64, y [][]float64, err error) {
	lenx := len(ix)
	leny := len(iy)

	if lenx < 1 || leny < 1 {
		return x, y, fmt.Errorf("expected input slices 'x' and 'y' to have at least one record along dimension (0)")
	}

	if lenx != leny {
		return x, y, fmt.Errorf("expected input slices 'x' and 'y' to have the same number of records along dimension (0): (%d) != (%d)", lenx, leny)
	}

	basex := len(ix[0])
	basey := len(iy[0])

	x = make([][]float64, lenx)
	y = make([][]float64, leny)

	for i := range lenx {
		ixi := ix[i]
		iyi := iy[i]

		lenxi := len(ixi)
		lenyi := len(iyi)

		if lenxi < 1 || lenyi < 1 {
			return x, y, fmt.Errorf("expected input slices 'x' and 'y' to have at least one record along dimension (1): got none at position (%d)", i)
		}

		if lenxi != basex {
			return x, y, fmt.Errorf("expected input slice 'x' to have equal length along every record in dimension (1): (%d) != (%d) at position (%d)", lenxi, basex, i)
		}

		if lenyi != basey {
			return x, y, fmt.Errorf("expected input slice 'y' to have equal length along every record in dimension (1): (%d) != (%d) at position (%d)", lenyi, basey, i)
		}

		x[i] = make([]float64, len(ixi))
		y[i] = make([]float64, len(iyi))

		copy(x[i], ixi)
		copy(y[i], iyi)
	}

	return x, y, nil
}
