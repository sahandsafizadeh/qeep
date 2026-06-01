package metrics

import (
	"fmt"
	"math"

	"github.com/sahandsafizadeh/qeep/tensor"
)

// Accuracy tracks correct predictions over accumulated batches.
type Accuracy struct {
	count      int
	correct    int
	oneHotMode bool
}

type AccuracyConfig struct {
	OneHotMode bool
}

const AccuracyDefaultOneHotMode = false

func NewAccuracy(conf *AccuracyConfig) *Accuracy {
	conf = toValidAccuracyConfig(conf)

	return &Accuracy{
		oneHotMode: conf.OneHotMode,
	}
}

func (c *Accuracy) Accumulate(yp tensor.Tensor, yt tensor.Tensor) (err error) {
	err = c.validateInputs(yp, yt)
	if err != nil {
		return fmt.Errorf("Accuracy input data validation failed: %w", err)
	}

	err = c.accumulate(yp, yt)
	if err != nil {
		return fmt.Errorf("Accuracy accumulate failed: %w", err)
	}

	return nil
}

func (c *Accuracy) accumulate(yp tensor.Tensor, yt tensor.Tensor) (err error) {
	if c.oneHotMode {
		yp, err = yp.Argmax(1)
		if err != nil {
			return err
		}

		yt, err = yt.Argmax(1)
		if err != nil {
			return err
		}
	} else {
		mid, err := c.toUntrackedFull(yp, 0.5)
		if err != nil {
			return err
		}

		yp, err = yp.Ge(mid)
		if err != nil {
			return err
		}

		yt, err = yt.Ge(mid)
		if err != nil {
			return err
		}
	}

	eq, err := yp.Eq(yt)
	if err != nil {
		return err
	}

	shape := eq.Shape()

	c.count += shape[0]
	c.correct += int(eq.Sum())

	return nil
}

func (c *Accuracy) Result() float64 {
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
		return fmt.Errorf("expected input tensors to have exactly two dimensions (batch, class)")
	}

	if shapep[0] != shapet[0] {
		return fmt.Errorf("expected input tensor sizes to match along batch dimension: (%d) != (%d)", shapep[0], shapet[0])
	}

	if shapep[1] != shapet[1] {
		return fmt.Errorf("expected input tensor sizes to match along class dimension: (%d) != (%d)", shapep[1], shapet[1])
	}

	if !c.oneHotMode && shapep[1] != 1 {
		return fmt.Errorf("expected input tensor sizes to be equal to (1) along class dimension when not in one-hot mode: got (%d)", shapep[1])
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

func toValidAccuracyConfig(iconf *AccuracyConfig) *AccuracyConfig {
	if iconf == nil {
		iconf = &AccuracyConfig{
			OneHotMode: AccuracyDefaultOneHotMode,
		}
	}

	conf := new(AccuracyConfig)
	*conf = *iconf

	return conf
}
