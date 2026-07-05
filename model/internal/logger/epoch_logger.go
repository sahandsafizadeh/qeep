package logger

import (
	"bytes"
	"fmt"
	"sort"
	"time"
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
		return el, fmt.Errorf("EpochLogger config data validation failed: %w", err)
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

	fmt.Print(el.getStartNextEpochLog())
}

func (el *EpochLogger) ProgressBatch() {
	el.curBatch++

	fmt.Print("\r")
	fmt.Print(el.getProgressBatchLog())
}

func (el *EpochLogger) FinishEpoch(l float64, vres map[string]float64) {
	el.epochEnd = time.Now()

	fmt.Print("\r")
	fmt.Println(el.getFinishEpochLog(l, vres))
}

func (el *EpochLogger) getStartNextEpochLog() string {
	return fmt.Sprintf(
		"%s%s%s",
		epoch(el.curEpoch, el.epochs),
		space(),
		progress(el.curBatch, el.batches),
	)
}

func (el *EpochLogger) getProgressBatchLog() string {
	return fmt.Sprintf(
		"%s%s%s",
		epoch(el.curEpoch, el.epochs),
		space(),
		progress(el.curBatch, el.batches),
	)
}

func (el *EpochLogger) getFinishEpochLog(l float64, vres map[string]float64) string {
	format := bytes.NewBuffer(make([]byte, 0, 15))
	args := make([]any, 0, 10)

	format.WriteString("%s%s%s%s%s")
	args = append(
		args,
		epoch(el.curEpoch, el.epochs),
		space(),
		duration(el.epochStart, el.epochEnd),
		space(),
		loss(l),
	)

	if len(vres) != 0 {
		format.WriteString("%s%s")
		args = append(
			args,
			space(),
			validation(vres),
		)
	}

	return fmt.Sprintf(format.String(), args...)
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

func loss(l float64) string {
	return fmt.Sprintf("Loss: %.4f", l)
}

func validation(result map[string]float64) string {
	keys := make([]string, 0, len(result))
	for key := range result {
		keys = append(keys, key)
	}

	sort.Strings(keys)

	buf := bytes.NewBuffer(nil)
	buf.WriteString("Validations: ")
	buf.WriteString("[")
	for i, key := range keys {
		entry := fmt.Sprintf("%q: %.2f", key, result[key])

		buf.WriteString(entry)
		if i != len(keys)-1 {
			buf.WriteString(", ")
		}
	}
	buf.WriteString("]")

	return buf.String()
}

func space() string {
	return ";   "
}

/* ----- helpers ----- */

func validate(epochs int, batches int) (err error) {
	if epochs <= 0 {
		return fmt.Errorf("expected the number of epochs to be positive: got (%d)", epochs)
	}

	if batches <= 0 {
		return fmt.Errorf("expected the number of batches to be positive: got (%d)", batches)
	}

	return nil
}
