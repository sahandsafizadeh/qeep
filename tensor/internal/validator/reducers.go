package validator

import "fmt"

func ValidateReducedDimAgainstDims(dim int, dims []int) (err error) {
	if !(0 <= dim && dim < len(dims)) {
		err = fmt.Errorf("expected dimension to be in range [0,%d): got (%d)", len(dims), dim)
		return
	}

	return nil
}
