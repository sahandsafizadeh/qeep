package validator

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor/internal/tensor"
)

func ValidateInputDims(dims []int) (err error) {
	if len(dims) > tensor.MaxDims {
		err = fmt.Errorf("expected at most (%d) dimensions: got (%d)", tensor.MaxDims, len(dims))
		return
	}

	for i, d := range dims {
		if d <= 0 {
			err = fmt.Errorf("expected positive dimension sizes: got (%d) at position (%d)", d, i)
			return
		}
	}

	return nil
}

func ValidateRandUParams(l, u float64) (err error) {
	if !(l < u) {
		err = fmt.Errorf("expected uniform random lower bound to be less than the upper bound: (%f) >= (%f)", l, u)
		return
	}

	return nil
}

func ValidateRandNParams(_, s float64) (err error) {
	if !(s > 0) {
		err = fmt.Errorf("expected normal random standard deviation to be positive: got (%f)", s)
		return
	}

	return nil
}

func ValidateInputDataDimUnity(data any) (err error) {
	zeroLenErr := fmt.Errorf("expected data to not have zero length along any dimension")
	dimUnityErr := fmt.Errorf("expected data to have have equal length along every dimension")

	switch v := data.(type) {
	case float64:
	case []float64:
		if len(v) == 0 {
			return zeroLenErr
		}

	case [][]float64:
		if len(v) == 0 {
			return zeroLenErr
		}

		dim := len(v[0])
		for _, sub := range v {
			err = ValidateInputDataDimUnity(sub)
			if err != nil {
				return
			}

			if len(sub) != dim {
				return dimUnityErr
			}
		}

	case [][][]float64:
		if len(v) == 0 {
			return zeroLenErr
		}

		dim := len(v[0])
		for _, sub := range v {
			err = ValidateInputDataDimUnity(sub)
			if err != nil {
				return
			}

			if len(sub) != dim {
				return dimUnityErr
			}
		}

	case [][][][]float64:
		if len(v) == 0 {
			return zeroLenErr
		}

		dim := len(v[0])
		for _, sub := range v {
			err = ValidateInputDataDimUnity(sub)
			if err != nil {
				return
			}

			if len(sub) != dim {
				return dimUnityErr
			}
		}

	default:
		panic("unreachable: compiler accepted input must be of type float64 | []float64 | [][]float64 | [][][]float64 | [][][][]float64")
	}

	return nil
}

func ValidateConcatTensorsDimsAlongDim(tsDims [][]int, dim int) (err error) {
	base := tsDims[0]
	for i, dims := range tsDims {
		if len(dims) == 0 {
			err = fmt.Errorf("scalar tensor can not be concatenated: got tensor (%d)", i)
			return
		}

		if len(dims) != len(base) {
			err = fmt.Errorf("expected tensors to have the same number of dimensions: (%d) != (%d) for tensor (%d)", len(dims), len(base), i)
			return
		}

		if !(0 <= dim && dim < len(base)) {
			err = fmt.Errorf("expected concat dimension to be in range [0,%d): got (%d)", len(base), dim)
			return
		}

		for j, d := range dims {
			if j == dim {
				continue
			}

			if d != base[j] {
				err = fmt.Errorf("expected tensor sizes to match in all dimensions except (%d): (%d) != (%d) for dimension (%d) for tensor (%d)", dim, d, base[j], j, i)
				return
			}
		}
	}

	return nil
}
