package layers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/tensor"
)

type BatchNorm struct {
	momentum float64
	eps      float64

	Beta       tensor.Tensor
	Gamma      tensor.Tensor
	MovingMean tensor.Tensor
	MovingVar  tensor.Tensor
}

type BatchNormConfig struct {
	Momentum float64
	Eps      float64
	Device   tensor.Device
}

const (
	BatchNormDefaultMomentum = 0.99
	BatchNormDefaultEps      = 1e-3
)

func NewBatchNorm(conf *BatchNormConfig) (c *BatchNorm, err error) {
	conf, err = toValidBatchNormConfig(conf)
	if err != nil {
		err = fmt.Errorf("BatchNorm config data validation failed: %w", err)
		return
	}

	c = &BatchNorm{
		momentum: conf.Momentum,
		eps:      conf.Eps,
	}

	c.Beta, err = tensor.Full(nil, 0., &tensor.Config{
		Device:    conf.Device,
		GradTrack: true,
	})
	if err != nil {
		return
	}

	c.Gamma, err = tensor.Full(nil, 1., &tensor.Config{
		Device:    conf.Device,
		GradTrack: true,
	})
	if err != nil {
		return
	}

	c.MovingMean, err = tensor.Full(nil, 0., &tensor.Config{
		Device:    conf.Device,
		GradTrack: false,
	})
	if err != nil {
		return
	}

	c.MovingVar, err = tensor.Full(nil, 1., &tensor.Config{
		Device:    conf.Device,
		GradTrack: false,
	})
	if err != nil {
		return
	}

	return c, nil
}

func (c *BatchNorm) Weights() []Weight {
	return []Weight{
		{
			Value:     &c.Beta,
			Trainable: true,
		},
		{
			Value:     &c.Gamma,
			Trainable: true,
		},
		{
			Value:     &c.MovingMean,
			Trainable: false, // not tracked
		},
		{
			Value:     &c.MovingVar,
			Trainable: false, // not tracked
		},
	}
}

func (c *BatchNorm) Forward(xs ...tensor.Tensor) (y tensor.Tensor, err error) {
	x, err := c.toValidInputs(xs)
	if err != nil {
		err = fmt.Errorf("BatchNorm input data validation failed: %w", err)
		return
	}

	return c.forward(x)
}

func (c *BatchNorm) forward(x tensor.Tensor) (y tensor.Tensor, err error) {
	// always normalize the last dim
	// take average across ALL of the instances for the last dim
	// meaning that if (..., F) is the shape of the input tensor, (F,) will be the shape of mean and variance

	var mean, _var tensor.Tensor

	/* ----- mean/var preparation ----- */

	if !x.GradientTracked() {
		mean = c.MovingMean
		_var = c.MovingVar
	} else {
		var fx tensor.Tensor

		fx, err = flattenToNormDim(x)
		if err != nil {
			return
		}

		mean, err = fx.MeanAlong(0)
		if err != nil {
			return
		}

		_var, err = fx.VarAlong(0)
		if err != nil {
			return
		}

		momentum := c.momentum
		movingMean := c.MovingMean
		movingVar := c.MovingVar

		mm1 := movingMean.Scale(momentum)
		mv1 := movingVar.Scale(momentum)
		mm2 := mean.Scale(1 - momentum)
		mv2 := _var.Scale(1 - momentum)

		c.MovingMean, err = mm1.Add(mm2)
		if err != nil {
			return
		}

		c.MovingVar, err = mv1.Add(mv2)
		if err != nil {
			return
		}
	}

	/* ----- normalization ----- */

	dev := _var.Device()
	dims := _var.Shape()

	epsilone, err := tensor.Full(dims, c.eps, &tensor.Config{
		Device:    dev,
		GradTrack: false,
	})
	if err != nil {
		return
	}

	n, err := x.Sub(mean)
	if err != nil {
		return
	}

	d, err := _var.Add(epsilone)
	if err != nil {
		return
	}

	d = d.Pow(0.5)

	y, err = n.Div(d)
	if err != nil {
		return
	}

	y, err = y.Mul(c.Gamma)
	if err != nil {
		return
	}

	y, err = y.Add(c.Beta)
	if err != nil {
		return
	}

	return y, nil
}

func flattenToNormDim(x tensor.Tensor) (y tensor.Tensor, err error) {
	dims := x.Shape()
	ndim := len(dims) - 1 // normalization dim

	flattened := 1
	for i := range ndim {
		flattened *= dims[i]
	}

	y, err = x.Reshape([]int{flattened, dims[ndim]})
	if err != nil {
		return
	}

	return y, nil
}

/* ----- helpers ----- */

func (c *BatchNorm) toValidInputs(xs []tensor.Tensor) (x tensor.Tensor, err error) {
	if len(xs) != 1 {
		err = fmt.Errorf("expected exactly one input tensor: got (%d)", len(xs))
		return
	}

	x = xs[0]

	shape := x.Shape()

	if len(shape) < 2 {
		err = fmt.Errorf("expected input tensor to have at least two dimensions (batch, ..., feature): got (%d)", len(shape))
		return
	}

	return x, nil
}

func toValidBatchNormConfig(iconf *BatchNormConfig) (conf *BatchNormConfig, err error) {
	if iconf == nil {
		err = fmt.Errorf("expected config not to be nil")
		return
	}

	conf = new(BatchNormConfig)
	*conf = *iconf

	if conf.Momentum < 0 {
		err = fmt.Errorf("expected 'Momentum' not to be negative: got (%f)", conf.Momentum)
		return
	}

	if conf.Eps <= 0 {
		err = fmt.Errorf("expected 'Eps' to be positive: got (%f)", conf.Eps)
		return
	}

	if conf.Device == 0 {
		conf.Device = tensor.CPU
	}

	return conf, nil
}
