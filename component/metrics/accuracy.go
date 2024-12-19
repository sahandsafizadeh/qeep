package metrics

import (
	"fmt"
	"math"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type Accuracy struct {
	count      int
	correct    int
	oneHotMode bool
}

type AccuracyConfig struct {
	OneHotMode bool
}

const accuracyDefaultOneHotMode = false

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

func (c *Accuracy) Result() (result float64, err error) {
	if c.count == 0 {
		return math.NaN(), nil
	}

	return float64(c.correct) / float64(c.count), nil
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

func toValidAccuracyConfig(iconf *AccuracyConfig) (conf *AccuracyConfig) {
	if iconf == nil {
		iconf = &AccuracyConfig{
			OneHotMode: accuracyDefaultOneHotMode,
		}
	}

	conf = new(AccuracyConfig)
	*conf = *iconf

	return conf
}
