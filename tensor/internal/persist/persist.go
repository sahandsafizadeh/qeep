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
	f, err := os.Create(path) // #nosec G304
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

	err = writeTensorArchive(f, s)
	if err != nil {
		return err
	}

	return nil
}

func Load(path string) (s *core.Snapshot, err error) {
	f, err := os.Open(path) // #nosec G304
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

func writeTensorArchive(f *os.File, s *core.Snapshot) (err error) {
	zw := zip.NewWriter(f)

	defer func() {
		if dferr := zw.Close(); err == nil && dferr != nil {
			err = dferr
		}
	}()

	err = writeBinaryFile(zw, metaFileName, toint64s(s.Dims))
	if err != nil {
		return fmt.Errorf("failed to write %q file: %w", metaFileName, err)
	}

	err = writeBinaryFile(zw, dataFileName, s.Data)
	if err != nil {
		return fmt.Errorf("failed to write %q file: %w", dataFileName, err)
	}

	return nil
}

func readTensorArchive(f *os.File) (s *core.Snapshot, err error) {
	info, err := f.Stat()
	if err != nil {
		return s, err
	}

	zr, err := zip.NewReader(f, info.Size())
	if err != nil {
		return s, err
	}

	meta, err := readBinaryFile[int64](zr, metaFileName)
	if err != nil {
		return s, fmt.Errorf("failed to read %q file: %w", metaFileName, err)
	}

	data, err := readBinaryFile[float64](zr, dataFileName)
	if err != nil {
		return s, fmt.Errorf("failed to read %q file: %w", dataFileName, err)
	}

	return &core.Snapshot{
		Dims: toints(meta),
		Data: data,
	}, nil
}

func writeBinaryFile[T int64 | float64](zw *zip.Writer, name string, content []T) (err error) {
	w, err := zw.Create(name)
	if err != nil {
		return err
	}

	err = binary.Write(w, binary.LittleEndian, content)
	if err != nil {
		return err
	}

	return nil
}

func readBinaryFile[T int64 | float64](zr *zip.Reader, name string) (content []T, err error) {
	f, err := zr.Open(name)
	if err != nil {
		return content, err
	}

	defer func() {
		if dferr := f.Close(); err == nil && dferr != nil {
			err = dferr
		}
	}()

	info, err := f.Stat()
	if err != nil {
		return content, err
	}

	var elem T
	elemsize := binary.Size(elem)
	unitsize := int64(elemsize)
	filesize := info.Size()

	if filesize%unitsize != 0 {
		return content, fmt.Errorf("corrupt file in archive: file size (%d) is not a multiple of expected unit (%d)", filesize, unitsize)
	}

	content = make([]T, filesize/unitsize)

	err = binary.Read(f, binary.LittleEndian, content)
	if err != nil {
		return content, err
	}

	return content, nil
}

/* ----- helpers ----- */

func toint64s(dims []int) (res []int64) {
	res = make([]int64, len(dims))
	for i, d := range dims {
		res[i] = int64(d)
	}

	return res
}

func toints(dims []int64) (res []int) {
	res = make([]int, len(dims))
	for i, d := range dims {
		res[i] = int(d)
	}

	return res
}
