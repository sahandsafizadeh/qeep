package validator

import "fmt"

func ValidateTransposeDims(dims []int32) (err error) {
	if len(dims) < 2 {
		err = fmt.Errorf("expected tensor to have at least (2) dimensions for transpose: got (%d)", len(dims))
		return
	}

	return nil
}

func ValidateReshapeSourceDimsAgainstTargetDims(srcDims, dstDims []int32) (err error) {
	srcElems := dimsToNumElems(srcDims)
	dstElems := dimsToNumElems(dstDims)

	if dstElems != srcElems {
		err = fmt.Errorf("expected number of elements in source and target tensors to match: (%d) != (%d)", srcElems, dstElems)
		return
	}

	return nil
}

func ValidateUnSqueezeDimAgainstDims(dim int32, dims []int32) (err error) {
	if !(0 <= dim && dim <= int32(len(dims))) {
		err = fmt.Errorf("expected dimension to be in range [0,%d]: got (%d)", len(dims), dim)
		return
	}

	return nil
}

func ValidateSqueezeDimAgainstDims(dim int32, dims []int32) (err error) {
	if !(0 <= dim && dim < int32(len(dims))) {
		err = fmt.Errorf("expected dimension to be in range [0,%d): got (%d)", len(dims), dim)
		return
	}

	if dims[dim] != 1 {
		err = fmt.Errorf("expected squeeze dimension to be (1): got (%d)", dims[dim])
		return
	}

	return nil
}

func ValidateFlattenDimAgainstDims(dim int32, dims []int32) (err error) {
	if !(0 <= dim && dim < int32(len(dims))) {
		err = fmt.Errorf("expected dimension to be in range [0,%d): got (%d)", len(dims), dim)
		return
	}

	return nil
}

func ValidateBroadcastSourceDimsAgainstTargetDims(srcDims, dstDims []int32) (err error) {
	i := len(srcDims)
	j := len(dstDims)

	if i > j {
		err = fmt.Errorf("expected number of dimensions in source tensor to be less than or equal to that of target shape: (%d) > (%d)", i, j)
		return
	}

	for i > 0 {
		i--
		j--

		if !(srcDims[i] == dstDims[j] || srcDims[i] == 1) {
			err = fmt.Errorf("expected target shape to be (%d) or source size to be (1) at dimension (%d): got (%d)", srcDims[i], j, dstDims[j])
			return
		}
	}

	return nil
}

/* ----- helpers ----- */

func dimsToNumElems(dims []int32) (elems int64) {
	elems = int64(1)
	for _, dim := range dims {
		elems *= int64(dim)
	}

	return elems
}
