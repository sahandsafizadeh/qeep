package batchgens_test

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/model/batchgens"
	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestSimple(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		x := [][]float64{
			{0., 1.},
			{1., 2.},
			{2., 3.},
			{3., 4.},
			{4., 5.},
		}
		y := [][]float64{
			{1, 0},
			{0, 1},
			{1, 0},
			{0, 1},
			{1, 0},
		}

		/* ------------------------------ */

		batchgen, err := batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
			BatchSize: 1,
			Shuffle:   false,
			Device:    conf.Device,
		})
		if err != nil {
			t.Fatal(err)
		}

		count := batchgen.Count()
		if count != 5 {
			t.Fatalf("expected batch generator's count value to be (5): got (%d)", count)
		}

		hasNext := batchgen.HasNext()
		if !hasNext {
			t.Fatalf("expected batch generator to have next batch")
		}

		actx, acty, err := batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		expx, err := tensor.Of([][]float64{
			{0., 1.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		expy, err := tensor.Of([][]float64{
			{1, 0},
		}, conf)
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

		/* --------------- */

		actx, acty, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		expx, err = tensor.Of([][]float64{
			{1., 2.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		expy, err = tensor.Of([][]float64{
			{0, 1},
		}, conf)
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

		/* --------------- */

		actx, acty, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		expx, err = tensor.Of([][]float64{
			{2., 3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		expy, err = tensor.Of([][]float64{
			{1, 0},
		}, conf)
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

		/* --------------- */

		actx, acty, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		expx, err = tensor.Of([][]float64{
			{3., 4.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		expy, err = tensor.Of([][]float64{
			{0, 1},
		}, conf)
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

		/* --------------- */

		actx, acty, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		expx, err = tensor.Of([][]float64{
			{4., 5.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		expy, err = tensor.Of([][]float64{
			{1, 0},
		}, conf)
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

		count = batchgen.Count()
		if count != 5 {
			t.Fatalf("expected batch generator's count value to be (5): got (%d)", count)
		}

		hasNext = batchgen.HasNext()
		if hasNext {
			t.Fatalf("expected batch generator not to have next batch")
		}

		/* ------------------------------ */

		batchgen, err = batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
			BatchSize: 2,
			Shuffle:   false,
			Device:    conf.Device,
		})
		if err != nil {
			t.Fatal(err)
		}

		count = batchgen.Count()
		if count != 3 {
			t.Fatalf("expected batch generator's count value to be (3): got (%d)", count)
		}

		hasNext = batchgen.HasNext()
		if !hasNext {
			t.Fatalf("expected batch generator to have next batch")
		}

		actx, acty, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		expx, err = tensor.Of([][]float64{
			{0., 1.},
			{1., 2.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		expy, err = tensor.Of([][]float64{
			{1, 0},
			{0, 1},
		}, conf)
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

		/* --------------- */

		actx, acty, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		expx, err = tensor.Of([][]float64{
			{2., 3.},
			{3., 4.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		expy, err = tensor.Of([][]float64{
			{1, 0},
			{0, 1},
		}, conf)
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

		/* --------------- */

		actx, acty, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		expx, err = tensor.Of([][]float64{
			{4., 5.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		expy, err = tensor.Of([][]float64{
			{1, 0},
		}, conf)
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

		count = batchgen.Count()
		if count != 3 {
			t.Fatalf("expected batch generator's count value to be (3): got (%d)", count)
		}

		hasNext = batchgen.HasNext()
		if hasNext {
			t.Fatalf("expected batch generator not to have next batch")
		}

		/* ------------------------------ */

		batchgen, err = batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
			BatchSize: 3,
			Shuffle:   false,
			Device:    conf.Device,
		})
		if err != nil {
			t.Fatal(err)
		}

		count = batchgen.Count()
		if count != 2 {
			t.Fatalf("expected batch generator's count value to be (2): got (%d)", count)
		}

		hasNext = batchgen.HasNext()
		if !hasNext {
			t.Fatalf("expected batch generator to have next batch")
		}

		actx, acty, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		expx, err = tensor.Of([][]float64{
			{0., 1.},
			{1., 2.},
			{2., 3.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		expy, err = tensor.Of([][]float64{
			{1, 0},
			{0, 1},
			{1, 0},
		}, conf)
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

		/* --------------- */

		actx, acty, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		expx, err = tensor.Of([][]float64{
			{3., 4.},
			{4., 5.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		expy, err = tensor.Of([][]float64{
			{0, 1},
			{1, 0},
		}, conf)
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

		count = batchgen.Count()
		if count != 2 {
			t.Fatalf("expected batch generator's count value to be (2): got (%d)", count)
		}

		hasNext = batchgen.HasNext()
		if hasNext {
			t.Fatalf("expected batch generator not to have next batch")
		}

		/* ------------------------------ */

		batchgen, err = batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
			BatchSize: 4,
			Shuffle:   false,
			Device:    conf.Device,
		})
		if err != nil {
			t.Fatal(err)
		}

		count = batchgen.Count()
		if count != 2 {
			t.Fatalf("expected batch generator's count value to be (2): got (%d)", count)
		}

		hasNext = batchgen.HasNext()
		if !hasNext {
			t.Fatalf("expected batch generator to have next batch")
		}

		actx, acty, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		expx, err = tensor.Of([][]float64{
			{0., 1.},
			{1., 2.},
			{2., 3.},
			{3., 4.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		expy, err = tensor.Of([][]float64{
			{1, 0},
			{0, 1},
			{1, 0},
			{0, 1},
		}, conf)
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

		/* --------------- */

		actx, acty, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		expx, err = tensor.Of([][]float64{
			{4., 5.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		expy, err = tensor.Of([][]float64{
			{1, 0},
		}, conf)
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

		count = batchgen.Count()
		if count != 2 {
			t.Fatalf("expected batch generator's count value to be (2): got (%d)", count)
		}

		hasNext = batchgen.HasNext()
		if hasNext {
			t.Fatalf("expected batch generator not to have next batch")
		}

		/* ------------------------------ */

		batchgen, err = batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
			BatchSize: 5,
			Shuffle:   false,
			Device:    conf.Device,
		})
		if err != nil {
			t.Fatal(err)
		}

		count = batchgen.Count()
		if count != 1 {
			t.Fatalf("expected batch generator's count value to be (1): got (%d)", count)
		}

		hasNext = batchgen.HasNext()
		if !hasNext {
			t.Fatalf("expected batch generator to have next batch")
		}

		actx, acty, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		expx, err = tensor.Of([][]float64{
			{0., 1.},
			{1., 2.},
			{2., 3.},
			{3., 4.},
			{4., 5.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		expy, err = tensor.Of([][]float64{
			{1, 0},
			{0, 1},
			{1, 0},
			{0, 1},
			{1, 0},
		}, conf)
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

		count = batchgen.Count()
		if count != 1 {
			t.Fatalf("expected batch generator's count value to be (1): got (%d)", count)
		}

		hasNext = batchgen.HasNext()
		if hasNext {
			t.Fatalf("expected batch generator not to have next batch")
		}

		/* ------------------------------ */

		batchgen, err = batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
			BatchSize: 6,
			Shuffle:   false,
			Device:    conf.Device,
		})
		if err != nil {
			t.Fatal(err)
		}

		count = batchgen.Count()
		if count != 1 {
			t.Fatalf("expected batch generator's count value to be (1): got (%d)", count)
		}

		hasNext = batchgen.HasNext()
		if !hasNext {
			t.Fatalf("expected batch generator to have next batch")
		}

		actx, acty, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		expx, err = tensor.Of([][]float64{
			{0., 1.},
			{1., 2.},
			{2., 3.},
			{3., 4.},
			{4., 5.},
		}, conf)
		if err != nil {
			t.Fatal(err)
		}

		expy, err = tensor.Of([][]float64{
			{1, 0},
			{0, 1},
			{1, 0},
			{0, 1},
			{1, 0},
		}, conf)
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

		count = batchgen.Count()
		if count != 1 {
			t.Fatalf("expected batch generator's count value to be (1): got (%d)", count)
		}

		hasNext = batchgen.HasNext()
		if hasNext {
			t.Fatalf("expected batch generator not to have next batch")
		}

		/* ------------------------------ */

		batchgen, err = batchgens.NewSimple(x, y, &batchgens.SimpleConfig{
			BatchSize: 5,
			Shuffle:   true,
			Device:    conf.Device,
		})
		if err != nil {
			t.Fatal(err)
		}

		_, _, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		hasNext = batchgen.HasNext()
		if hasNext {
			t.Fatalf("expected batch generator not to have next batch")
		}

		batchgen.Reset()

		hasNext = batchgen.HasNext()
		if !hasNext {
			t.Fatalf("expected batch generator to have next batch")
		}

		_, _, err = batchgen.NextBatch()
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

	})
}

func TestValidationSimple(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		/* ------------------------------ */

		_, err := batchgens.NewSimple([][]float64{{1.}}, [][]float64{{0}}, nil)
		if err == nil {
			t.Fatalf("expected error because of nil input config")
		} else if err.Error() != "Simple config data validation failed: expected config not to be nil" {
			t.Fatal("unexpected error message returned")
		}

		_, err = batchgens.NewSimple([][]float64{{1.}}, [][]float64{{0}},
			&batchgens.SimpleConfig{
				BatchSize: 0,
			})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'BatchSize'")
		} else if err.Error() != "Simple config data validation failed: expected 'BatchSize' to be positive: got (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = batchgens.NewSimple([][]float64{{1.}}, [][]float64{{0}},
			&batchgens.SimpleConfig{
				BatchSize: -1,
			})
		if err == nil {
			t.Fatalf("expected error because of non-positive 'BatchSize'")
		} else if err.Error() != "Simple config data validation failed: expected 'BatchSize' to be positive: got (-1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = batchgens.NewSimple([][]float64{}, [][]float64{{0}},
			&batchgens.SimpleConfig{
				BatchSize: 2,
			})
		if err == nil {
			t.Fatalf("expected error because of input slices not having at least one record along dimension (0)")
		} else if err.Error() != "Simple input data validation failed: expected input slices 'x' and 'y' to have at least one record along dimension (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = batchgens.NewSimple([][]float64{{1.}}, [][]float64{},
			&batchgens.SimpleConfig{
				BatchSize: 2,
			})
		if err == nil {
			t.Fatalf("expected error because of input slices not having at least one record along dimension (0)")
		} else if err.Error() != "Simple input data validation failed: expected input slices 'x' and 'y' to have at least one record along dimension (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = batchgens.NewSimple([][]float64{{1.}, {1.}}, [][]float64{{0}},
			&batchgens.SimpleConfig{
				BatchSize: 2,
			})
		if err == nil {
			t.Fatalf("expected error because of input slices not having the same number of records along dimension (0)")
		} else if err.Error() != "Simple input data validation failed: expected input slices 'x' and 'y' to have the same number of records along dimension (0): (2) != (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = batchgens.NewSimple([][]float64{{1.}, {}}, [][]float64{{0}, {0}},
			&batchgens.SimpleConfig{
				BatchSize: 2,
			})
		if err == nil {
			t.Fatalf("expected error because of input slices not having at least one record along dimension (1)")
		} else if err.Error() != "Simple input data validation failed: expected input slices 'x' and 'y' to have at least one record along dimension (1): got none at position (1)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = batchgens.NewSimple([][]float64{{1.}, {1.}}, [][]float64{{}, {0}},
			&batchgens.SimpleConfig{
				BatchSize: 2,
			})
		if err == nil {
			t.Fatalf("expected error because of input slices not having at least one record along dimension (1)")
		} else if err.Error() != "Simple input data validation failed: expected input slices 'x' and 'y' to have at least one record along dimension (1): got none at position (0)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = batchgens.NewSimple([][]float64{{1.}, {1.}, {1., 1.}}, [][]float64{{0}, {0}, {0}},
			&batchgens.SimpleConfig{
				BatchSize: 2,
			})
		if err == nil {
			t.Fatalf("expected error because of input slice 'x' not having equal length along every record in dimension (1)")
		} else if err.Error() != "Simple input data validation failed: expected input slice 'x' to have equal length along every record in dimension (1): (2) != (1) at position (2)" {
			t.Fatal("unexpected error message returned")
		}

		_, err = batchgens.NewSimple([][]float64{{1.}, {1.}, {1.}}, [][]float64{{0}, {0}, {0, 0}},
			&batchgens.SimpleConfig{
				BatchSize: 2,
			})
		if err == nil {
			t.Fatalf("expected error because of input slice 'y' not having equal length along every record in dimension (1)")
		} else if err.Error() != "Simple input data validation failed: expected input slice 'y' to have equal length along every record in dimension (1): (2) != (1) at position (2)" {
			t.Fatal("unexpected error message returned")
		}

		/* ------------------------------ */

		batchgen, err := batchgens.NewSimple([][]float64{{1.}}, [][]float64{{0}},
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

		/* ------------------------------ */

	})
}
