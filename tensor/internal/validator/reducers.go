package validator

import "fmt"

func ValidateReducedDimAgainstDims(dim int32, dims []int32) (err error) {
	if !(0 <= dim && dim < int32(len(dims))) {
		err = fmt.Errorf("expected dimension to be in range [0,%d): got (%d)", len(dims), dim)
		return
	}

	return nil
}
