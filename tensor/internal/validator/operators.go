package validator

import "fmt"

func ValidateBinaryFuncDimsMatch(dims1, dims2 []int32) (err error) {
	if len(dims1) != len(dims2) {
		err = fmt.Errorf("expected number of dimensions to match: (%d) != (%d)", len(dims1), len(dims2))
		return
	}

	for i := 0; i < len(dims1); i++ {
		if dims1[i] != dims2[i] {
			err = fmt.Errorf("expected sizes to match at dimension (%d): (%d) != (%d)", i, dims1[i], dims2[i])
			return
		}
	}

	return nil
}

func ValidateDotProductDims(dims1, dims2 []int32) (err error) {
	ldt1 := len(dims1)
	ldt2 := len(dims2)

	if ldt1 < 1 || ldt2 < 1 {
		err = fmt.Errorf("expected tensors to have at least (1) dimension for dot product: got (%d) and (%d)", ldt1, ldt2)
		return
	}

	if dims1[ldt1-1] != dims2[ldt2-1] {
		err = fmt.Errorf("expected sizes to match at last dimensions: (%d) != (%d)", dims1[ldt1-1], dims2[ldt2-1])
		return
	}

	return nil
}

func ValidateMatMulDims(dims1, dims2 []int32) (err error) {
	ldt1 := len(dims1)
	ldt2 := len(dims2)

	if ldt1 < 2 || ldt2 < 2 {
		err = fmt.Errorf("expected tensors to have at least (2) dimensions for matrix multiplication: got (%d) and (%d)", ldt1, ldt2)
		return
	}

	if dims1[ldt1-1] != dims2[ldt2-2] {
		err = fmt.Errorf("expected dimension (%d) of first tensor to be equal to dimension (%d) of second tensor for matrix multiplication: (%d) != (%d)",
			ldt1-1, ldt2-2, dims1[ldt1-1], dims2[ldt2-2])
		return
	}

	return nil
}
