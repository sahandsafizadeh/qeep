package persist

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/sahandsafizadeh/qeep/tensor/internal/core"
)

func Save(s *core.Snapshot, path string) (err error) {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create tensor file: %w", err)
	}

	defer func() {
		cerr := f.Close()
		if err == nil && cerr != nil {
			err = fmt.Errorf("failed to close tensor file: %w", cerr)
		}

		// a partially written file is not loadable: leave nothing behind
		if err != nil {
			os.Remove(path)
		}
	}()

	err = writeArchive(f, s)
	if err != nil {
		return err
	}

	return nil
}

func writeArchive(w io.Writer, s *core.Snapshot) (err error) {
	zw := zip.NewWriter(w)

	err = writeBinaryFile(zw, dataFileName, s.Data)
	if err != nil {
		return err
	}

	err = writeBinaryFile(zw, metaFileName, toInt64s(s.Dims))
	if err != nil {
		return err
	}

	err = zw.Close()
	if err != nil {
		return fmt.Errorf("failed to close tensor archive: %w", err)
	}

	return nil
}

func writeBinaryFile(zw *zip.Writer, name string, data any) (err error) {
	w, err := zw.Create(name)
	if err != nil {
		return fmt.Errorf("failed to create %q file of tensor archive: %w", name, err)
	}

	err = binary.Write(w, byteOrder, data)
	if err != nil {
		return fmt.Errorf("failed to write %q file of tensor archive: %w", name, err)
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
