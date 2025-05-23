package validator

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
)

func ValidateAtIndexAgainstDims(index []int, dims []int) (err error) {
	if len(index) != len(dims) {
		err = fmt.Errorf("expected index length to be equal to the number of dimensions: (%d) != (%d)", len(index), len(dims))
		return
	}

	for i, idx := range index {
		if idx < 0 || idx >= dims[i] {
			err = fmt.Errorf("expected index to be in range [0,%d) at dimension (%d): got (%d)", dims[i], i, idx)
			return
		}
	}

	return nil
}

func ValidateSliceIndexAgainstDims(index []tensor.Range, dims []int) (err error) {
	if len(index) > len(dims) {
		err = fmt.Errorf("expected index length to be smaller than or equal to the number of dimensions: (%d) > (%d)", len(index), len(dims))
		return
	}

	for i, idx := range index {
		// ignore special case
		if idx.From == 0 && idx.To == 0 {
			continue
		}

		if idx.From >= idx.To {
			err = fmt.Errorf("expected range 'From' to be smaller than 'To' except for special both (0) case (fetchAll): (%d) >= (%d) at dimension (%d)", idx.From, idx.To, i)
			return
		}

		if idx.From < 0 || idx.From >= dims[i] ||
			idx.To < 1 || idx.To >= dims[i]+1 {

			if idx.To == idx.From+1 {
				err = fmt.Errorf("expected index to be in range [0,%d) at dimension (%d): got (%d)", dims[i], i, idx.From)
				return
			} else {
				err = fmt.Errorf("expected index to fall in range [0,%d] at dimension (%d): got [%d,%d)", dims[i], i, idx.From, idx.To)
				return
			}
		}
	}

	return nil
}

func ValidatePatchIndexAgainstDims(index []tensor.Range, srcDims, dstDims []int) (err error) {
	if len(srcDims) != len(dstDims) {
		err = fmt.Errorf("expected number of dimensions to match among source and target tensors: (%d) != (%d)", len(srcDims), len(dstDims))
		return
	}

	for i := 0; i < len(srcDims); i++ {
		if srcDims[i] > dstDims[i] {
			err = fmt.Errorf("expected source tensor size not to exceed that of target tensor at dimension (%d): (%d) > (%d)", i, srcDims[i], dstDims[i])
			return
		}
	}

	err = ValidateSliceIndexAgainstDims(index, dstDims)
	if err != nil {
		err = fmt.Errorf("index incompatible with target tensor: %w", err)
		return
	}

	for i, idx := range index {
		// ignore special case
		if idx.From == 0 && idx.To == 0 {
			continue
		}

		if (idx.To - idx.From) != srcDims[i] {
			err = fmt.Errorf("expected index to exactly cover source tensor at dimension (%d): #[%d,%d) != (%d)", i, idx.From, idx.To, srcDims[i])
			return
		}
	}

	return nil
}
