package persist

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/sahandsafizadeh/qeep/tensor/internal/core"
)

// maxDimSize keeps dimension sizes read from file within a platform independent range,
// so that the number of elements they imply can be computed without overflow.
const maxDimSize = math.MaxInt32

// Load reads a tensor snapshot from the zip archive at path, as written by Save.
// The ".qeep" extension is appended to path if it's missing.
func Load(path string) (s *core.Snapshot, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open tensor file: %w", err)
	}

	defer func() {
		cerr := f.Close()
		if err == nil && cerr != nil {
			err = fmt.Errorf("failed to close tensor file: %w", cerr)
		}
	}()

	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to read tensor file information: %w", err)
	}

	s, err = readArchive(f, info.Size())
	if err != nil {
		return nil, err
	}

	return s, nil
}

func readArchive(r io.ReaderAt, size int64) (s *core.Snapshot, err error) {
	zr, err := zip.NewReader(r, size)
	if err != nil {
		return nil, fmt.Errorf("failed to open tensor archive: %w", err)
	}

	data, err := readBinaryFile[float64](zr, dataFileName)
	if err != nil {
		return nil, err
	}

	meta, err := readBinaryFile[int64](zr, metaFileName)
	if err != nil {
		return nil, err
	}

	return toSnapshot(data, meta)
}

func readBinaryFile[T int64 | float64](zr *zip.Reader, name string) (data []T, err error) {
	f, err := zr.Open(name)
	if err != nil {
		return nil, fmt.Errorf("failed to open %q file of tensor archive: %w", name, err)
	}

	defer func() {
		cerr := f.Close()
		if err == nil && cerr != nil {
			err = fmt.Errorf("failed to close %q file of tensor archive: %w", name, cerr)
		}
	}()

	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to read %q file information of tensor archive: %w", name, err)
	}

	var elem T
	elemSize := int64(binary.Size(elem))

	size := info.Size()
	if size%elemSize != 0 {
		return nil, fmt.Errorf("corrupt %q file of tensor archive: size (%d) is not a multiple of (%d)", name, size, elemSize)
	}

	data = make([]T, size/elemSize)

	err = binary.Read(f, byteOrder, data)
	if err != nil {
		return nil, fmt.Errorf("failed to read %q file of tensor archive: %w", name, err)
	}

	return data, nil
}

func toSnapshot(data []float64, meta []int64) (s *core.Snapshot, err error) {
	dims, err := toDims(meta)
	if err != nil {
		return nil, err
	}

	err = validateDataLenAgainstDims(len(data), dims)
	if err != nil {
		return nil, err
	}

	return &core.Snapshot{
		Dims: dims,
		Data: data,
	}, nil
}

func toDims(ms []int64) (dims []int, err error) {
	dims = make([]int, len(ms))
	for i, m := range ms {
		if m <= 0 || m > maxDimSize {
			return nil, fmt.Errorf("corrupt tensor archive: invalid dimension size (%d) at position (%d)", m, i)
		}

		dims[i] = int(m)
	}

	return dims, nil
}

func validateDataLenAgainstDims(n int, dims []int) (err error) {
	count := 1
	for _, d := range dims {
		count *= d

		// dimensions are capped: leaving early keeps the product from overflowing
		if count > n {
			break
		}
	}

	if count != n {
		return fmt.Errorf("corrupt tensor archive: dimensions %v do not match the number of elements (%d)", dims, n)
	}

	return nil
}
