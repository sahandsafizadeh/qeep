package validator

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/core"
)

func ValidateAtIndexAgainstDims(index []int, dims []int) (err error) {
	if len(index) != len(dims) {
		return fmt.Errorf("expected index length to be equal to the number of dimensions: (%d) != (%d)", len(index), len(dims))
	}

	for i, idx := range index {
		if idx < 0 || idx >= dims[i] {
			return fmt.Errorf("expected index to be in range [0,%d) at dimension (%d): got (%d)", dims[i], i, idx)
		}
	}

	return nil
}

func ValidateSliceIndexAgainstDims(index []core.Range, dims []int) (err error) {
	if len(index) > len(dims) {
		return fmt.Errorf("expected index length to be smaller than or equal to the number of dimensions: (%d) > (%d)", len(index), len(dims))
	}

	for i, idx := range index {
		// ignore special case
		if idx.From == 0 && idx.To == 0 {
			continue
		}

		if idx.From >= idx.To {
			return fmt.Errorf("expected range 'From' to be smaller than 'To' except for special both (0) case (fetchAll): (%d) >= (%d) at dimension (%d)", idx.From, idx.To, i)
		}

		if idx.From < 0 || idx.From >= dims[i] ||
			idx.To < 1 || idx.To >= dims[i]+1 {

			if idx.To == idx.From+1 {
				return fmt.Errorf("expected index to be in range [0,%d) at dimension (%d): got (%d)", dims[i], i, idx.From)
			} else {
				return fmt.Errorf("expected index to fall in range [0,%d] at dimension (%d): got [%d,%d)", dims[i], i, idx.From, idx.To)
			}
		}
	}

	return nil
}
