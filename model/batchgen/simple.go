package batchgen

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Simple struct {
	x         tensor.Tensor
	y         tensor.Tensor
	batchSize int
	length    int
	index     int
}

func NewSimple(x tensor.Tensor, y tensor.Tensor, batchSize int) (bg *Simple, err error) {
	err = validateDataShape(x, y)
	if err != nil {
		return
	}

	err = validateBatchSize(batchSize)
	if err != nil {
		return
	}

	return &Simple{
		x:         x,
		y:         y,
		batchSize: batchSize,
		length:    x.Shape()[0],
		index:     0,
	}, nil
}

func (bg *Simple) Reset() {
	bg.index = 0
}

func (bg *Simple) Count() (count int) {
	return bg.length/bg.batchSize + 1
}

func (bg *Simple) HasNext() (ok bool) {
	return bg.index < bg.length
}

func (bg *Simple) NextBatch() (xs []tensor.Tensor, y tensor.Tensor, err error) {
	i := bg.index
	length := bg.length
	batchSize := bg.batchSize

	from := i * batchSize
	to := (i + 1) * batchSize
	if to > length {
		to = length
	}

	x, err := bg.x.Slice([]tensor.Range{{From: from, To: to}})
	if err != nil {
		return
	}

	y, err = bg.y.Slice([]tensor.Range{{From: from, To: to}})
	if err != nil {
		return
	}

	bg.index += bg.batchSize

	return []tensor.Tensor{x}, y, nil
}

/* ----- helpers ----- */

func validateDataShape(x tensor.Tensor, y tensor.Tensor) (err error) {
	shapex := x.Shape()
	shapey := y.Shape()

	if len(shapex) == 0 {
		err = fmt.Errorf("expected tensor x not to be scalar and have at least one dimension for batch")
		return
	}

	if len(shapey) == 0 {
		err = fmt.Errorf("expected tensor y not to be scalar and have at least one dimension for batch")
		return
	}

	sbx := shapex[0]
	sby := shapey[0]

	if sbx != sby {
		err = fmt.Errorf("expected tensors x and y to have the same size along batch dimension: (%d) != (%d)", sbx, sby)
		return
	}

	return nil
}

func validateBatchSize(batchSize int) (err error) {
	if batchSize <= 0 {
		err = fmt.Errorf("expected input batch size to be positive: (%d) <= 0", batchSize)
		return
	}

	return nil
}
