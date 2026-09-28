package persist

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"os"

	"github.com/sahandsafizadeh/qeep/tensor/internal/core"
)

const (
	metaFileName = "meta"
	dataFileName = "data"
)

func Save(s *core.Snapshot, path string) (err error) {
	f, err := os.Create(path)
	if err != nil {
		return err
	}

	defer func() {
		if dferr := f.Close(); err == nil && dferr != nil {
			err = dferr
		}
		if err != nil {
			_ = os.Remove(path)
		}
	}()

	err = writeTensorArchive(s, f)
	if err != nil {
		return err
	}

	return nil
}

func Load(path string) (s *core.Snapshot, err error) {
	f, err := os.Open(path)
	if err != nil {
		return s, err
	}

	defer func() {
		if dferr := f.Close(); err == nil && dferr != nil {
			err = dferr
		}
	}()

	s, err = readTensorArchive(f)
	if err != nil {
		return s, err
	}

	return s, nil
}

func writeTensorArchive(s *core.Snapshot, f *os.File) (err error) {
	zw := zip.NewWriter(f)

	defer func() {
		if dferr := zw.Close(); err == nil && dferr != nil {
			err = dferr
		}
	}()

	err = writeBinaryFile(toint8s(s.Dims), metaFileName, zw)
	if err != nil {
		return fmt.Errorf("failed to write %q file: %w", metaFileName, err)
	}

	err = writeBinaryFile(s.Data, dataFileName, zw)
	if err != nil {
		return fmt.Errorf("failed to write %q file: %w", dataFileName, err)
	}

	return nil
}

func toInt64s(dims []int) (ds []int64) {
	ds = make([]int64, len(dims))
	for i, d := range dims {
		ds[i] = int64(d)
	}

	return ds
}

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

	meta, err := readBinaryFile[int64](zr, metaFileName)
	if err != nil {
		return nil, err
	}

	data, err := readBinaryFile[float64](zr, dataFileName)
	if err != nil {
		return nil, err
	}

	return &core.Snapshot{
		Dims: toints(meta),
		Data: data,
	}, nil
}

func writeBinaryFile[T int8 | float64](data []T, name string, zw *zip.Writer) (err error) {
	w, err := zw.Create(name)
	if err != nil {
		return err
	}

	err = binary.Write(w, binary.LittleEndian, data)
	if err != nil {
		return err
	}

	return nil
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

	err = binary.Read(f, binary.LittleEndian, data)
	if err != nil {
		return nil, fmt.Errorf("failed to read %q file of tensor archive: %w", name, err)
	}

	return data, nil
}

func toint8s(dims []int) (res []int8) {
	res = make([]int8, len(dims))
	for i, d := range dims {
		res[i] = int8(d)
	}

	return res
}

func toints(dims []int8) (res []int) {
	res = make([]int, len(dims))
	for i, d := range dims {
		res[i] = int(d)
	}

	return res
}
