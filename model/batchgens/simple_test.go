package batchgens_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/model/batchgens"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestSimple(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("NewSimple(5 records, BatchSize=1, no shuffle) / iterating all 5 batches / Count=5 and each record returned individually in order", func(t *testing.T) {
			x := [][]float64{
				{0, 1},
				{1, 2},
				{2, 3},
				{3, 4},
				{4, 5},
			}
			y := [][]float64{
				{1, 0},
				{0, 1},
				{1, 0},
				{0, 1},
				{1, 0},
			}

			batchgen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
				BatchSize: 1,
				Shuffle:   false,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			expectedBatches := []struct {
				x [][]float64
				y [][]float64
			}{
				{x: [][]float64{{0, 1}}, y: [][]float64{{1, 0}}},
				{x: [][]float64{{1, 2}}, y: [][]float64{{0, 1}}},
				{x: [][]float64{{2, 3}}, y: [][]float64{{1, 0}}},
				{x: [][]float64{{3, 4}}, y: [][]float64{{0, 1}}},
				{x: [][]float64{{4, 5}}, y: [][]float64{{1, 0}}},
			}
			if count := batchgen.Count(); count != 5 {
				t.Fatalf("expected batch generator's count value to be (5): got (%d)", count)
			}

			if !batchgen.HasNext() {
				t.Fatalf("expected batch generator to have next batch")
			}
			for _, exp := range expectedBatches {
				assertNextBatch(t, batchgen, exp.x, exp.y, dev)
			}
			if batchgen.HasNext() {
				t.Fatalf("expected batch generator not to have next batch")
			}
		})

		t.Run("NewSimple(5 records, BatchSize=2, no shuffle) / iterating all 3 batches / Count=3, pairs of records with remainder of 1", func(t *testing.T) {
			x := [][]float64{
				{0, 1},
				{1, 2},
				{2, 3},
				{3, 4},
				{4, 5},
			}
			y := [][]float64{
				{1, 0},
				{0, 1},
				{1, 0},
				{0, 1},
				{1, 0},
			}

			batchgen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
				BatchSize: 2,
				Shuffle:   false,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			expectedBatches := []struct {
				x [][]float64
				y [][]float64
			}{
				{x: [][]float64{{0, 1}, {1, 2}}, y: [][]float64{{1, 0}, {0, 1}}},
				{x: [][]float64{{2, 3}, {3, 4}}, y: [][]float64{{1, 0}, {0, 1}}},
				{x: [][]float64{{4, 5}}, y: [][]float64{{1, 0}}},
			}
			if count := batchgen.Count(); count != 3 {
				t.Fatalf("expected batch generator's count value to be (3): got (%d)", count)
			}

			if !batchgen.HasNext() {
				t.Fatalf("expected batch generator to have next batch")
			}
			for _, exp := range expectedBatches {
				assertNextBatch(t, batchgen, exp.x, exp.y, dev)
			}
			if batchgen.HasNext() {
				t.Fatalf("expected batch generator not to have next batch")
			}
		})

		t.Run("NewSimple(5 records, BatchSize=3, no shuffle) / iterating all 2 batches / Count=2, first batch of 3 and remainder of 2", func(t *testing.T) {
			x := [][]float64{
				{0, 1},
				{1, 2},
				{2, 3},
				{3, 4},
				{4, 5},
			}
			y := [][]float64{
				{1, 0},
				{0, 1},
				{1, 0},
				{0, 1},
				{1, 0},
			}

			batchgen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
				BatchSize: 3,
				Shuffle:   false,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			expectedBatches := []struct {
				x [][]float64
				y [][]float64
			}{
				{x: [][]float64{{0, 1}, {1, 2}, {2, 3}}, y: [][]float64{{1, 0}, {0, 1}, {1, 0}}},
				{x: [][]float64{{3, 4}, {4, 5}}, y: [][]float64{{0, 1}, {1, 0}}},
			}
			if count := batchgen.Count(); count != 2 {
				t.Fatalf("expected batch generator's count value to be (2): got (%d)", count)
			}

			if !batchgen.HasNext() {
				t.Fatalf("expected batch generator to have next batch")
			}
			for _, exp := range expectedBatches {
				assertNextBatch(t, batchgen, exp.x, exp.y, dev)
			}
			if batchgen.HasNext() {
				t.Fatalf("expected batch generator not to have next batch")
			}
		})

		t.Run("NewSimple(5 records, BatchSize=4, no shuffle) / iterating all 2 batches / Count=2, first batch of 4 and remainder of 1", func(t *testing.T) {
			x := [][]float64{
				{0, 1},
				{1, 2},
				{2, 3},
				{3, 4},
				{4, 5},
			}
			y := [][]float64{
				{1, 0},
				{0, 1},
				{1, 0},
				{0, 1},
				{1, 0},
			}

			batchgen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
				BatchSize: 4,
				Shuffle:   false,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			expectedBatches := []struct {
				x [][]float64
				y [][]float64
			}{
				{x: [][]float64{{0, 1}, {1, 2}, {2, 3}, {3, 4}}, y: [][]float64{{1, 0}, {0, 1}, {1, 0}, {0, 1}}},
				{x: [][]float64{{4, 5}}, y: [][]float64{{1, 0}}},
			}
			if count := batchgen.Count(); count != 2 {
				t.Fatalf("expected batch generator's count value to be (2): got (%d)", count)
			}

			if !batchgen.HasNext() {
				t.Fatalf("expected batch generator to have next batch")
			}
			for _, exp := range expectedBatches {
				assertNextBatch(t, batchgen, exp.x, exp.y, dev)
			}
			if batchgen.HasNext() {
				t.Fatalf("expected batch generator not to have next batch")
			}
		})

		t.Run("NewSimple(5 records, BatchSize=5, no shuffle) / iterating 1 batch / Count=1 and all 5 records returned in single batch", func(t *testing.T) {
			x := [][]float64{
				{0, 1},
				{1, 2},
				{2, 3},
				{3, 4},
				{4, 5},
			}
			y := [][]float64{
				{1, 0},
				{0, 1},
				{1, 0},
				{0, 1},
				{1, 0},
			}

			batchgen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
				BatchSize: 5,
				Shuffle:   false,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			expectedBatches := []struct {
				x [][]float64
				y [][]float64
			}{
				{
					x: [][]float64{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}},
					y: [][]float64{{1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}},
				},
			}
			if count := batchgen.Count(); count != 1 {
				t.Fatalf("expected batch generator's count value to be (1): got (%d)", count)
			}

			if !batchgen.HasNext() {
				t.Fatalf("expected batch generator to have next batch")
			}
			for _, exp := range expectedBatches {
				assertNextBatch(t, batchgen, exp.x, exp.y, dev)
			}
			if batchgen.HasNext() {
				t.Fatalf("expected batch generator not to have next batch")
			}
		})

		t.Run("NewSimple(5 records, BatchSize=6 exceeds data size, no shuffle) / iterating 1 batch / Count=1 and all 5 records returned as single batch", func(t *testing.T) {
			x := [][]float64{
				{0, 1},
				{1, 2},
				{2, 3},
				{3, 4},
				{4, 5},
			}
			y := [][]float64{
				{1, 0},
				{0, 1},
				{1, 0},
				{0, 1},
				{1, 0},
			}

			batchgen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
				BatchSize: 6,
				Shuffle:   false,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			expectedBatches := []struct {
				x [][]float64
				y [][]float64
			}{
				{
					x: [][]float64{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}},
					y: [][]float64{{1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}},
				},
			}
			if count := batchgen.Count(); count != 1 {
				t.Fatalf("expected batch generator's count value to be (1): got (%d)", count)
			}

			if !batchgen.HasNext() {
				t.Fatalf("expected batch generator to have next batch")
			}
			for _, exp := range expectedBatches {
				assertNextBatch(t, batchgen, exp.x, exp.y, dev)
			}
			if batchgen.HasNext() {
				t.Fatalf("expected batch generator not to have next batch")
			}
		})

		t.Run("NewSimple(5 records, BatchSize=5, shuffle=true) / NextBatch then Reset then NextBatch / HasNext returns true after Reset", func(t *testing.T) {
			x := [][]float64{
				{0, 1},
				{1, 2},
				{2, 3},
				{3, 4},
				{4, 5},
			}
			y := [][]float64{
				{1, 0},
				{0, 1},
				{1, 0},
				{0, 1},
				{1, 0},
			}

			batchgen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
				BatchSize: 5,
				Shuffle:   true,
				Device:    dev,
			})
			if err != nil {
				t.Fatal(err)
			}

			if _, _, err = batchgen.NextBatch(); err != nil {
				t.Fatal(err)
			} else if batchgen.HasNext() {
				t.Fatalf("expected batch generator not to have next batch")
			}

			batchgen.Reset()

			if !batchgen.HasNext() {
				t.Fatalf("expected batch generator to have next batch")
			} else if _, _, err = batchgen.NextBatch(); err != nil {
				t.Fatal(err)
			}
		})

		// ============================== validations ==============================

		t.Run("NewSimple with nil config / returns error: config not to be nil", func(t *testing.T) {
			_, err := batchgens.NewSimple([][]float64{{1}}, [][]float64{{0}}, nil)
			if err == nil {
				t.Fatalf("expected error because of nil input config")
			} else if err.Error() != "Simple config data validation failed: expected config not to be nil" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewSimple with BatchSize=0 / returns error: BatchSize not positive", func(t *testing.T) {
			_, err := batchgens.NewSimple([][]float64{{1}}, [][]float64{{0}},
				&batchgens.SimpleConfig{BatchSize: 0})
			if err == nil {
				t.Fatalf("expected error because of non-positive 'BatchSize'")
			} else if err.Error() != "Simple config data validation failed: expected 'BatchSize' to be positive: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewSimple with BatchSize=-1 / returns error: BatchSize not positive", func(t *testing.T) {
			_, err := batchgens.NewSimple([][]float64{{1}}, [][]float64{{0}},
				&batchgens.SimpleConfig{BatchSize: -1})
			if err == nil {
				t.Fatalf("expected error because of non-positive 'BatchSize'")
			} else if err.Error() != "Simple config data validation failed: expected 'BatchSize' to be positive: got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewSimple with empty x / returns error: x and y must have at least one record along dimension 0", func(t *testing.T) {
			_, err := batchgens.NewSimple([][]float64{}, [][]float64{{0}},
				&batchgens.SimpleConfig{BatchSize: 2})
			if err == nil {
				t.Fatalf("expected error because of input slices not having at least one record along dimension (0)")
			} else if err.Error() != "Simple input data validation failed: expected input slices 'x' and 'y' to have at least one record along dimension (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewSimple with empty y / returns error: x and y must have at least one record along dimension 0", func(t *testing.T) {
			_, err := batchgens.NewSimple([][]float64{{1}}, [][]float64{},
				&batchgens.SimpleConfig{BatchSize: 2})
			if err == nil {
				t.Fatalf("expected error because of input slices not having at least one record along dimension (0)")
			} else if err.Error() != "Simple input data validation failed: expected input slices 'x' and 'y' to have at least one record along dimension (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewSimple with x having 2 records and y having 1 / returns error: record count mismatch along dimension 0", func(t *testing.T) {
			_, err := batchgens.NewSimple([][]float64{{1}, {1}}, [][]float64{{0}},
				&batchgens.SimpleConfig{BatchSize: 2})
			if err == nil {
				t.Fatalf("expected error because of input slices not having the same number of records along dimension (0)")
			} else if err.Error() != "Simple input data validation failed: expected input slices 'x' and 'y' to have the same number of records along dimension (0): (2) != (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewSimple with x having empty row at position 1 / returns error: no records along dimension 1 at that position", func(t *testing.T) {
			_, err := batchgens.NewSimple([][]float64{{1}, {}}, [][]float64{{0}, {0}},
				&batchgens.SimpleConfig{BatchSize: 2})
			if err == nil {
				t.Fatalf("expected error because of input slices not having at least one record along dimension (1)")
			} else if err.Error() != "Simple input data validation failed: expected input slices 'x' and 'y' to have at least one record along dimension (1): got none at position (1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewSimple with y having empty row at position 0 / returns error: no records along dimension 1 at that position", func(t *testing.T) {
			_, err := batchgens.NewSimple([][]float64{{1}, {1}}, [][]float64{{}, {0}},
				&batchgens.SimpleConfig{BatchSize: 2})
			if err == nil {
				t.Fatalf("expected error because of input slices not having at least one record along dimension (1)")
			} else if err.Error() != "Simple input data validation failed: expected input slices 'x' and 'y' to have at least one record along dimension (1): got none at position (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewSimple with x rows of inconsistent length / returns error: x rows must have equal length along dimension 1", func(t *testing.T) {
			_, err := batchgens.NewSimple([][]float64{{1}, {1}, {1, 1}}, [][]float64{{0}, {0}, {0}},
				&batchgens.SimpleConfig{BatchSize: 2})
			if err == nil {
				t.Fatalf("expected error because of input slice 'x' not having equal length along every record in dimension (1)")
			} else if err.Error() != "Simple input data validation failed: expected input slice 'x' to have equal length along every record in dimension (1): (2) != (1) at position (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewSimple with y rows of inconsistent length / returns error: y rows must have equal length along dimension 1", func(t *testing.T) {
			_, err := batchgens.NewSimple([][]float64{{1}, {1}, {1}}, [][]float64{{0}, {0}, {0, 0}},
				&batchgens.SimpleConfig{BatchSize: 2})
			if err == nil {
				t.Fatalf("expected error because of input slice 'y' not having equal length along every record in dimension (1)")
			} else if err.Error() != "Simple input data validation failed: expected input slice 'y' to have equal length along every record in dimension (1): (2) != (1) at position (2)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewSimple batchgen after exhaustion / NextBatch() / returns error: next batch does not exist", func(t *testing.T) {
			batchgen, err := batchgens.NewSimple([][]float64{{1}}, [][]float64{{0}},
				&batchgens.SimpleConfig{
					BatchSize: 1,
					Shuffle:   true,
				})
			if err != nil {
				t.Fatal(err)
			}

			_, _, err = batchgen.NextBatch()
			if err != nil {
				t.Fatal(err)
			}

			_, _, err = batchgen.NextBatch()
			if err == nil {
				t.Fatalf("expected error because of non-existing next batch")
			} else if err.Error() != "Simple state validation failed: expected next batch to exist" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

/* ----- helpers ----- */

func assertNextBatch(
	t *testing.T,
	batchgen *batchgens.Simple,
	expX [][]float64,
	expY [][]float64,
	dev tensor.Device,
) {
	t.Helper()

	actx, acty, err := batchgen.NextBatch()
	if err != nil {
		t.Fatal(err)
	}

	expx, err := tensor.Of(expX, &tensor.Config{Device: dev})
	if err != nil {
		t.Fatal(err)
	}
	expy, err := tensor.Of(expY, &tensor.Config{Device: dev})
	if err != nil {
		t.Fatal(err)
	}

	if eq, err := actx[0].Equals(expx); err != nil {
		t.Fatal(err)
	} else if !eq {
		t.Fatalf("expected tensors to be equal")
	}
	if eq, err := acty.Equals(expy); err != nil {
		t.Fatal(err)
	} else if !eq {
		t.Fatalf("expected tensors to be equal")
	}
}
