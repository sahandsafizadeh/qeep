package metrics

import (
	"fmt"
	"math"

	"github.com/sahandsafizadeh/qeep/tensor"
)

// Accuracy tracks correct predictions over accumulated batches.
// Call Result() after accumulation to get the fraction of correct predictions.
type Accuracy struct {
	count      int
	correct    int
	oneHotMode bool
}

// AccuracyConfig controls how predictions are compared to targets.
// OneHotMode=true: use argmax for multi-class (one-hot encoded).
// OneHotMode=false: use 0.5 threshold for binary classification.
type AccuracyConfig struct {
	OneHotMode bool
}

const AccuracyDefaultOneHotMode = false

// NewAccuracy creates an Accuracy metric. conf may be nil; then OneHotMode defaults to false.
func NewAccuracy(conf *AccuracyConfig) (c *Accuracy) {
	conf = toValidAccuracyConfig(conf)

	return &Accuracy{
		oneHotMode: conf.OneHotMode,
	}
}

func (c *Accuracy) Accumulate(yp tensor.Tensor, yt tensor.Tensor) (err error) {
	err = c.validateInputs(yp, yt)
	if err != nil {
		err = fmt.Errorf("Accuracy input data validation failed: %w", err)
		return
	}

	if c.oneHotMode {
		yp, err = yp.Argmax(1)
		if err != nil {
			return
		}

		yt, err = yt.Argmax(1)
		if err != nil {
			return
		}
	} else {
		var mid tensor.Tensor

		mid, err = c.toUntrackedFull(yp, 0.5)
		if err != nil {
			return
		}

		yp, err = yp.Ge(mid)
		if err != nil {
			return
		}

		yt, err = yt.Ge(mid)
		if err != nil {
			return
		}
	}

	eq, err := yp.Eq(yt)
	if err != nil {
		return
	}

	shape := eq.Shape()

	c.count += shape[0]
	c.correct += int(eq.Sum())

	return nil
}

func (c *Accuracy) Result() (result float64) {
	if c.count == 0 {
		return math.NaN()
	}

	return float64(c.correct) / float64(c.count)
}

/* ----- helpers ----- */

func (c *Accuracy) validateInputs(yp tensor.Tensor, yt tensor.Tensor) (err error) {
	shapep := yp.Shape()
	shapet := yt.Shape()

	if len(shapep) != 2 || len(shapet) != 2 {
		err = fmt.Errorf("expected input tensors to have exactly two dimensions (batch, class)")
		return
	}

	if shapep[0] != shapet[0] {
		err = fmt.Errorf("expected input tensor sizes to match along batch dimension: (%d) != (%d)", shapep[0], shapet[0])
		return
	}

	if shapep[1] != shapet[1] {
		err = fmt.Errorf("expected input tensor sizes to match along class dimension: (%d) != (%d)", shapep[1], shapet[1])
		return
	}

	if !c.oneHotMode && shapep[1] != 1 {
		err = fmt.Errorf("expected input tensor sizes to be equal to (1) along class dimension when not in one-hot mode: got (%d)", shapep[1])
		return
	}

	return nil
}

func (c *Accuracy) toUntrackedFull(x tensor.Tensor, value float64) (y tensor.Tensor, err error) {
	dev := x.Device()
	dims := x.Shape()

	return tensor.Full(dims, value, &tensor.Config{
		Device:    dev,
		GradTrack: false,
	})
}

func toValidAccuracyConfig(iconf *AccuracyConfig) (conf *AccuracyConfig) {
	if iconf == nil {
		iconf = &AccuracyConfig{
			OneHotMode: AccuracyDefaultOneHotMode,
		}
	}

	conf = new(AccuracyConfig)
	*conf = *iconf

	return conf
}
