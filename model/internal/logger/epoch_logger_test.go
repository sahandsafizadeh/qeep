package logger

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

/* ------------------------------ White Box Tested ------------------------------ */

func TestEpochLogger(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		loss, err := tensor.TensorOf(1.5, conf)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		const (
			epochs  = 5
			batches = 10
		)

		epochLogger, err := NewEpochLogger(epochs, batches)
		if err != nil {
			t.Fatal(err)
		}

		/* ------------------------------ */

		if epochLogger.epochs != epochs {
			t.Fatalf("expected epochLogger's epochs to be (%d): got (%d)", epochs, epochLogger.epochs)
		}
		if epochLogger.batches != batches {
			t.Fatalf("expected epochLogger's batches to be (%d): got (%d)", batches, epochLogger.batches)
		}

		/* ------------------------------ */

		for e := range epochs {
			epochLogger.StartNextEpoch()

			if epochLogger.curEpoch != e+1 {
				t.Fatalf("expected epochLogger's current epoch to be (%d): got (%d)", e+1, epochLogger.curEpoch)
			}
			if epochLogger.curBatch != 0 {
				t.Fatalf("expected epochLogger's current batch to be (%d): got (%d)", 0, epochLogger.curBatch)
			}

			for b := range batches {
				epochLogger.ProgressBatch()

				if epochLogger.curEpoch != e+1 {
					t.Fatalf("expected epochLogger's current epoch to be (%d): got (%d)", e+1, epochLogger.curEpoch)
				}
				if epochLogger.curBatch != b+1 {
					t.Fatalf("expected epochLogger's current batch to be (%d): got (%d)", b+1, epochLogger.curBatch)
				}
			}

			epochLogger.FinishEpoch(loss)
		}

		if epochLogger.curEpoch != epochs {
			t.Fatalf("expected epochLogger's current epoch to be (%d): got (%d)", epochs, epochLogger.curEpoch)
		}
		if epochLogger.curBatch != batches {
			t.Fatalf("expected epochLogger's current batch to be (%d): got (%d)", batches, epochLogger.curBatch)
		}

		/* ------------------------------ */

	})

}

func TestValidationEpochLogger(t *testing.T) {

	/* ------------------------------ */

	_, err := NewEpochLogger(0, 1)
	if err == nil {
		t.Fatalf("expected error because of non-positive number of epochs")
	} else if err.Error() != "EpochLogger config data validation failed: expected the number of epochs to be positive: got (0)" {
		t.Fatal("unexpected error message returned")
	}

	_, err = NewEpochLogger(-1, 1)
	if err == nil {
		t.Fatalf("expected error because of non-positive number of epochs")
	} else if err.Error() != "EpochLogger config data validation failed: expected the number of epochs to be positive: got (-1)" {
		t.Fatal("unexpected error message returned")
	}

	_, err = NewEpochLogger(1, 0)
	if err == nil {
		t.Fatalf("expected error because of non-positive number of batches")
	} else if err.Error() != "EpochLogger config data validation failed: expected the number of batches to be positive: got (0)" {
		t.Fatal("unexpected error message returned")
	}

	_, err = NewEpochLogger(1, -1)
	if err == nil {
		t.Fatalf("expected error because of non-positive number of batches")
	} else if err.Error() != "EpochLogger config data validation failed: expected the number of batches to be positive: got (-1)" {
		t.Fatal("unexpected error message returned")
	}

	/* ------------------------------ */

}
