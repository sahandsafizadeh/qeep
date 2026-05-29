package validator

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
	"github.com/sahandsafizadeh/qeep/tensor/internal/util"
)

func ValidateTransposeDims(dims []int) error {
	if len(dims) < 2 {
		return fmt.Errorf("expected tensor to have at least (2) dimensions for transpose: got (%d)", len(dims))
	}

	return nil
}

func ValidateReshapeSourceDimsAgainstTargetDims(srcDims, dstDims []int) error {
	srcElems := util.DimsToNumElems(srcDims)
	dstElems := util.DimsToNumElems(dstDims)

	if dstElems != srcElems {
		return fmt.Errorf("expected number of elements in source and target tensors to match: (%d) != (%d)", srcElems, dstElems)
	}

	return nil
}

func ValidateUnSqueezeDimAgainstDims(dim int, dims []int) error {
	if len(dims) == tensor.MaxDims {
		return fmt.Errorf("operation causes tensor to exceed maximum (%d) dimensions", tensor.MaxDims)
	}

	if dim < 0 || dim > len(dims) {
		return fmt.Errorf("expected dimension to be in range [0,%d]: got (%d)", len(dims), dim)
	}

	return nil
}

func ValidateSqueezeDimAgainstDims(dim int, dims []int) error {
	if dim < 0 || dim >= len(dims) {
		return fmt.Errorf("expected dimension to be in range [0,%d): got (%d)", len(dims), dim)
	}

	if dims[dim] != 1 {
		return fmt.Errorf("expected squeeze dimension to be (1): got (%d)", dims[dim])
	}

	return nil
}

func ValidateFlattenDimAgainstDims(dim int, dims []int) error {
	if dim < 0 || dim >= len(dims) {
		return fmt.Errorf("expected dimension to be in range [0,%d): got (%d)", len(dims), dim)
	}

	return nil
}

func ValidateBroadcastSourceDimsAgainstTargetDims(srcDims, dstDims []int) error {
	i := len(srcDims)
	j := len(dstDims)

	if i > j {
		return fmt.Errorf("expected number of dimensions in source tensor to be less than or equal to that of target shape: (%d) > (%d)", i, j)
	}

	for i > 0 {
		i--
		j--

		if srcDims[i] != dstDims[j] && srcDims[i] != 1 {
			return fmt.Errorf("expected target shape to be (%d) or source size to be (1) at dimension (%d): got shape (%d)", srcDims[i], j, dstDims[j])
		}
	}

	return nil
}
