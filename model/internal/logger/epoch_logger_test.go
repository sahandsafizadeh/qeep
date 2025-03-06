package logger

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestEpochLogger(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		conf := &tensor.Config{Device: dev}

		/* ------------------------------ */

		loggerTest := func(
			epochs int,
			batches int,
			loss tensor.Tensor,
			vres map[string]float64,
			expStartLogs []string,
			expFinishLogs []string,
			expProgressLogs [][]string,
		) {
			epochLogger, err := NewEpochLogger(epochs, batches)
			if err != nil {
				t.Fatal(err)
			}

			for e := range epochs {
				epochLogger.StartNextEpoch()
				if epochLogger.getStartNextEpochLog() != expStartLogs[e] {
					t.Fatal("unexpected log message returned")
				}

				for b := range batches {
					epochLogger.ProgressBatch()
					if epochLogger.getProgressBatchLog() != expProgressLogs[e][b] {
						t.Fatal("unexpected log message returned")
					}
				}

				epochLogger.FinishEpoch(loss, vres)
				if epochLogger.getFinishEpochLog(loss, vres) != expFinishLogs[e] {
					t.Fatal("unexpected log message returned")
				}
			}
		}

		/* ------------------------------ */

		epochs := 1
		batches := 1
		loss, _ := tensor.TensorOf(1.5, conf)
		vres := map[string]float64{"MSE": 2.345, "Accuracy": 0.82, "dummy": 0.}

		expStartLogs := []string{
			"Epoch: 1/1;   Progress: 0%",
		}
		expFinishLogs := []string{
			"Epoch: 1/1;   Duration: 0s;   Loss: 1.500000;   Validation: map[Accuracy:0.82 MSE:2.345 dummy:0]",
		}
		expProgressLogs := [][]string{
			{
				"Epoch: 1/1;   Progress: 100%",
			},
		}

		loggerTest(
			epochs,
			batches,
			loss,
			vres,
			expStartLogs,
			expFinishLogs,
			expProgressLogs,
		)

		/* ------------------------------ */

		epochs = 1
		batches = 2
		loss, _ = tensor.TensorOf(0.55, conf)
		vres = map[string]float64{"MSE": 9.3, "Accuracy": 0.123}

		expStartLogs = []string{
			"Epoch: 1/1;   Progress: 0%",
		}
		expFinishLogs = []string{
			"Epoch: 1/1;   Duration: 0s;   Loss: 0.550000;   Validation: map[Accuracy:0.123 MSE:9.3]",
		}
		expProgressLogs = [][]string{
			{
				"Epoch: 1/1;   Progress: 50%",
				"Epoch: 1/1;   Progress: 100%",
			},
		}

		loggerTest(
			epochs,
			batches,
			loss,
			vres,
			expStartLogs,
			expFinishLogs,
			expProgressLogs,
		)

		/* ------------------------------ */

		epochs = 3
		batches = 1
		loss, _ = tensor.TensorOf(44.005, conf)
		vres = map[string]float64{"MSE": 0.56789}

		expStartLogs = []string{
			"Epoch: 1/3;   Progress: 0%",
			"Epoch: 2/3;   Progress: 0%",
			"Epoch: 3/3;   Progress: 0%",
		}
		expFinishLogs = []string{
			"Epoch: 1/3;   Duration: 0s;   Loss: 44.005000;   Validation: map[MSE:0.56789]",
			"Epoch: 2/3;   Duration: 0s;   Loss: 44.005000;   Validation: map[MSE:0.56789]",
			"Epoch: 3/3;   Duration: 0s;   Loss: 44.005000;   Validation: map[MSE:0.56789]",
		}
		expProgressLogs = [][]string{
			{
				"Epoch: 1/3;   Progress: 100%",
			},
			{
				"Epoch: 2/3;   Progress: 100%",
			},
			{
				"Epoch: 3/3;   Progress: 100%",
			},
		}

		loggerTest(
			epochs,
			batches,
			loss,
			vres,
			expStartLogs,
			expFinishLogs,
			expProgressLogs,
		)

		/* ------------------------------ */

		epochs = 2
		batches = 4
		loss, _ = tensor.TensorOf(456.1, conf)
		vres = nil

		expStartLogs = []string{
			"Epoch: 1/2;   Progress: 0%",
			"Epoch: 2/2;   Progress: 0%",
		}
		expFinishLogs = []string{
			"Epoch: 1/2;   Duration: 0s;   Loss: 456.100000",
			"Epoch: 2/2;   Duration: 0s;   Loss: 456.100000",
		}
		expProgressLogs = [][]string{
			{
				"Epoch: 1/2;   Progress: 25%",
				"Epoch: 1/2;   Progress: 50%",
				"Epoch: 1/2;   Progress: 75%",
				"Epoch: 1/2;   Progress: 100%",
			},
			{
				"Epoch: 2/2;   Progress: 25%",
				"Epoch: 2/2;   Progress: 50%",
				"Epoch: 2/2;   Progress: 75%",
				"Epoch: 2/2;   Progress: 100%",
			},
		}

		loggerTest(
			epochs,
			batches,
			loss,
			vres,
			expStartLogs,
			expFinishLogs,
			expProgressLogs,
		)

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
