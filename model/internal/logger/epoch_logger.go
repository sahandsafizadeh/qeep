package logger

import (
	"fmt"
	"time"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type EpochLogger struct {
	epochs     int
	batches    int
	curEpoch   int
	curBatch   int
	epochStart time.Time
	epochEnd   time.Time
}

func NewEpochLogger(epochs int, batches int) (el *EpochLogger, err error) {
	err = validate(epochs, batches)
	if err != nil {
		err = fmt.Errorf("EpochLogger config data validation failed: %w", err)
		return
	}

	return &EpochLogger{
		epochs:  epochs,
		batches: batches,
	}, nil
}

func (el *EpochLogger) StartNextEpoch() {
	el.curEpoch++
	el.curBatch = 0
	el.epochStart = time.Now()

	fmt.Printf(
		"%s%s%s",
		epoch(el.curEpoch, el.epochs),
		space(),
		progress(el.curBatch, el.batches),
	)
}

func (el *EpochLogger) ProgressBatch() {
	el.curBatch++

	fmt.Printf(
		"\r%s%s%s",
		epoch(el.curEpoch, el.epochs),
		space(),
		progress(el.curBatch, el.batches),
	)
}

func (el *EpochLogger) FinishEpoch(l tensor.Tensor) {
	el.epochEnd = time.Now()

	fmt.Printf(
		"\r%s%s%s%s%s\n",
		epoch(el.curEpoch, el.epochs),
		space(),
		duration(el.epochStart, el.epochEnd),
		space(),
		loss(l),
	)
}

func epoch(currentEpoch int, totalEpochs int) string {
	return fmt.Sprintf("Epoch: %d/%d", currentEpoch, totalEpochs)
}

func progress(currentBatch int, totalBatches int) string {
	ratio := float64(currentBatch) / float64(totalBatches)
	percentage := int(ratio * 100)
	return fmt.Sprintf("Progress: %d%%", percentage)
}

func duration(startTime time.Time, endTime time.Time) string {
	dur := endTime.Sub(startTime)
	dur = dur.Round(time.Second)
	return fmt.Sprintf("Duration: %s", dur)
}

func loss(l tensor.Tensor) string {
	loss, _ := l.At() // TEMPORARY: until tensors implement Stringer interface
	return fmt.Sprintf("Loss: %f", loss)
}

func space() string {
	return ";   "
}

/* ----- helpers ----- */

func validate(epochs int, batches int) (err error) {
	if epochs <= 0 {
		err = fmt.Errorf("expected the number of epochs to be positive: got (%d)", epochs)
		return
	}

	if batches <= 0 {
		err = fmt.Errorf("expected the number of batches to be positive: got (%d)", batches)
		return
	}

	return nil
}
