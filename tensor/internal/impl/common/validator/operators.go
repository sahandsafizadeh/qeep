package validator

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/core"
)

func ValidateBinaryFuncDimsMatch(dims1, dims2 []int) (err error) {
	if len(dims1) != len(dims2) {
		return fmt.Errorf("expected number of dimensions to match: (%d) != (%d)", len(dims1), len(dims2))
	}

	for i := range dims1 {
		if dims1[i] != dims2[i] {
			return fmt.Errorf("expected sizes to match at dimension (%d): (%d) != (%d)", i, dims1[i], dims2[i])
		}
	}

	return nil
}

func ValidateDotProductDims(dims1, dims2 []int) (err error) {
	ldt1 := len(dims1)
	ldt2 := len(dims2)

	if ldt1 < 1 || ldt2 < 1 {
		return fmt.Errorf("expected tensors to have at least (1) dimension for dot product: got (%d) and (%d)", ldt1, ldt2)
	}

	if dims1[ldt1-1] != dims2[ldt2-1] {
		return fmt.Errorf("expected sizes to match at last dimensions: (%d) != (%d)", dims1[ldt1-1], dims2[ldt2-1])
	}

	return nil
}

func ValidateMatMulDims(dims1, dims2 []int) (err error) {
	ldt1 := len(dims1)
	ldt2 := len(dims2)

	if ldt1 < 2 || ldt2 < 2 {
		return fmt.Errorf("expected tensors to have at least (2) dimensions for matrix multiplication: got (%d) and (%d)", ldt1, ldt2)
	}

	if dims1[ldt1-1] != dims2[ldt2-2] {
		return fmt.Errorf("expected dimension (%d) of first tensor to be equal to dimension (%d) of second tensor for matrix multiplication: (%d) != (%d)",
			ldt1-1, ldt2-2, dims1[ldt1-1], dims2[ldt2-2])
	}

	return nil
}

func ValidatePatchIndexAgainstDims(index []core.Range, srcDims, dstDims []int) (err error) {
	if len(srcDims) != len(dstDims) {
		return fmt.Errorf("expected number of dimensions to match among source and target tensors: (%d) != (%d)", len(srcDims), len(dstDims))
	}

	for i := range srcDims {
		if srcDims[i] > dstDims[i] {
			return fmt.Errorf("expected source tensor size not to exceed that of target tensor at dimension (%d): (%d) > (%d)", i, srcDims[i], dstDims[i])
		}
	}

	err = ValidateSliceIndexAgainstDims(index, dstDims)
	if err != nil {
		return fmt.Errorf("index incompatible with target tensor: %w", err)
	}

	for i, idx := range index {
		// ignore special case
		if idx.From == 0 && idx.To == 0 {
			continue
		}

		if (idx.To - idx.From) != srcDims[i] {
			return fmt.Errorf("expected index to exactly cover source tensor at dimension (%d): #[%d,%d) != (%d)", i, idx.From, idx.To, srcDims[i])
		}
	}

	return nil
}
