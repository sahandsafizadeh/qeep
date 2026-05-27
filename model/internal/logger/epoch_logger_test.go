package logger

import (
	"testing"

	"github.com/sahandsafizadeh/qeep/tensor"
)

func TestEpochLogger(t *testing.T) {
	tensor.RunTestLogicOnDevices(func(dev tensor.Device) {

		// ============================== main paths ==============================

		t.Run("1 epoch 1 batch loss=1.5 with validations / full run / logs match expected format", func(t *testing.T) {
			epochs := 1
			batches := 1
			loss, err := tensor.Of(1.5, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			vres := map[string]float64{"MSE": 2.34555, "Accuracy": 0.82, "dummy": 0.}
			expStartLogs := []string{
				`Epoch: 1/1;   Progress: 0%`,
			}
			expFinishLogs := []string{
				`Epoch: 1/1;   Duration: 0s;   Loss: 1.5000;   Validations: ["Accuracy": 0.82, "MSE": 2.35, "dummy": 0.00]`,
			}
			expProgressLogs := [][]string{
				{
					`Epoch: 1/1;   Progress: 100%`,
				},
			}

			runEpochLoggerTest(
				t,
				epochs,
				batches,
				loss,
				vres,
				expStartLogs,
				expFinishLogs,
				expProgressLogs,
			)
		})

		t.Run("1 epoch 2 batches loss=0.55 with validations / full run / logs match expected format", func(t *testing.T) {
			epochs := 1
			batches := 2
			loss, err := tensor.Of(0.55, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			vres := map[string]float64{"MSE": 9.3, "Accuracy": 0.123}
			expStartLogs := []string{
				`Epoch: 1/1;   Progress: 0%`,
			}
			expFinishLogs := []string{
				`Epoch: 1/1;   Duration: 0s;   Loss: 0.5500;   Validations: ["Accuracy": 0.12, "MSE": 9.30]`,
			}
			expProgressLogs := [][]string{
				{
					`Epoch: 1/1;   Progress: 50%`,
					`Epoch: 1/1;   Progress: 100%`,
				},
			}

			runEpochLoggerTest(
				t,
				epochs,
				batches,
				loss,
				vres,
				expStartLogs,
				expFinishLogs,
				expProgressLogs,
			)
		})

		t.Run("3 epochs 1 batch loss=44.005 with validations / full run / logs match expected format", func(t *testing.T) {
			epochs := 2
			batches := 4
			loss, err := tensor.Of(44.005, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			vres := map[string]float64{"MSE": 0.56789}
			expStartLogs := []string{
				`Epoch: 1/3;   Progress: 0%`,
				`Epoch: 2/3;   Progress: 0%`,
				`Epoch: 3/3;   Progress: 0%`,
			}
			expFinishLogs := []string{
				`Epoch: 1/3;   Duration: 0s;   Loss: 44.0050;   Validations: ["MSE": 0.57]`,
				`Epoch: 2/3;   Duration: 0s;   Loss: 44.0050;   Validations: ["MSE": 0.57]`,
				`Epoch: 3/3;   Duration: 0s;   Loss: 44.0050;   Validations: ["MSE": 0.57]`,
			}
			expProgressLogs := [][]string{
				{`Epoch: 1/3;   Progress: 100%`},
				{`Epoch: 2/3;   Progress: 100%`},
				{`Epoch: 3/3;   Progress: 100%`},
			}

			runEpochLoggerTest(
				t,
				epochs,
				batches,
				loss,
				vres,
				expStartLogs,
				expFinishLogs,
				expProgressLogs,
			)
		})

		t.Run("2 epochs 4 batches loss=456.1 no validations / full run / logs match expected format", func(t *testing.T) {
			epochs := 2
			batches := 4
			loss, err := tensor.Of(456.1, &tensor.Config{Device: dev})
			if err != nil {
				t.Fatal(err)
			}
			expStartLogs := []string{
				`Epoch: 1/2;   Progress: 0%`,
				`Epoch: 2/2;   Progress: 0%`,
			}
			expFinishLogs := []string{
				`Epoch: 1/2;   Duration: 0s;   Loss: 456.1000`,
				`Epoch: 2/2;   Duration: 0s;   Loss: 456.1000`,
			}
			expProgressLogs := [][]string{
				{
					`Epoch: 1/2;   Progress: 25%`,
					`Epoch: 1/2;   Progress: 50%`,
					`Epoch: 1/2;   Progress: 75%`,
					`Epoch: 1/2;   Progress: 100%`,
				},
				{
					`Epoch: 2/2;   Progress: 25%`,
					`Epoch: 2/2;   Progress: 50%`,
					`Epoch: 2/2;   Progress: 75%`,
					`Epoch: 2/2;   Progress: 100%`,
				},
			}

			runEpochLoggerTest(
				t,
				epochs,
				batches,
				loss,
				nil,
				expStartLogs,
				expFinishLogs,
				expProgressLogs,
			)
		})

		// ============================== validations ==============================

		t.Run("NewEpochLogger(0, 1) / returns error: non-positive number of epochs", func(t *testing.T) {
			_, err := NewEpochLogger(0, 1)
			if err == nil {
				t.Fatalf("expected error because of non-positive number of epochs")
			} else if err.Error() != "EpochLogger config data validation failed: expected the number of epochs to be positive: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewEpochLogger(-1, 1) / returns error: non-positive number of epochs", func(t *testing.T) {
			_, err := NewEpochLogger(-1, 1)
			if err == nil {
				t.Fatalf("expected error because of non-positive number of epochs")
			} else if err.Error() != "EpochLogger config data validation failed: expected the number of epochs to be positive: got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewEpochLogger(1, 0) / returns error: non-positive number of batches", func(t *testing.T) {
			_, err := NewEpochLogger(1, 0)
			if err == nil {
				t.Fatalf("expected error because of non-positive number of batches")
			} else if err.Error() != "EpochLogger config data validation failed: expected the number of batches to be positive: got (0)" {
				t.Fatal("unexpected error message returned")
			}
		})

		t.Run("NewEpochLogger(1, -1) / returns error: non-positive number of batches", func(t *testing.T) {
			_, err := NewEpochLogger(1, -1)
			if err == nil {
				t.Fatalf("expected error because of non-positive number of batches")
			} else if err.Error() != "EpochLogger config data validation failed: expected the number of batches to be positive: got (-1)" {
				t.Fatal("unexpected error message returned")
			}
		})
	})
}

func runEpochLoggerTest(
	t *testing.T,
	epochs int,
	batches int,
	loss tensor.Tensor,
	vres map[string]float64,
	expStartLogs []string,
	expFinishLogs []string,
	expProgressLogs [][]string,
) {
	t.Helper()

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
