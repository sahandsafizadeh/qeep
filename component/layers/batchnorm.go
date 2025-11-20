package layers

import (
	"fmt"

	"github.com/sahandsafizadeh/qeep/component/initializers"
	"github.com/sahandsafizadeh/qeep/tensor"
)

type BatchNorm struct {
	dim      int
	momentum float64
	eps      float64

	Beta       tensor.Tensor
	Gamma      tensor.Tensor
	MovingMean tensor.Tensor
	MovingVar  tensor.Tensor
}

type BatchNormConfig struct {
	Dim          int
	Momentum     float64
	Eps          float64
	Initializers map[string]Initializer
	Device       tensor.Device
}

const (
	BatchNormDefaultMomentum = 0.99
	BatchNormDefaultEps      = 1e-3
)

const (
	batchNormBetaKey       = "Beta"
	batchNormGammaKey      = "Gamma"
	batchNormMovingMeanKey = "MovingMean"
	batchNormMovingVarKey  = "MovingVar"
)

func NewBatchNorm(conf *BatchNormConfig) (c *BatchNorm, err error) {
	conf, err = toValidBatchNormConfig(conf)
	if err != nil {
		err = fmt.Errorf("BatchNorm config data validation failed: %w", err)
		return
	}

	// var (
	// 	win = conf.Initializers[fcWeightKey]
	// 	bin = conf.Initializers[fcBiasKey]
	// )

	// w, err := win.Init([]int{conf.Outputs}, conf.Device)
	// if err != nil {
	// 	return
	// }

	// b, err := bin.Init([]int{conf.Outputs}, conf.Device)
	// if err != nil {
	// 	return
	// }

	// err = validateInitializedWeights(w, b, conf)
	// if err != nil {
	// 	err = fmt.Errorf("FC initialized weight validation failed: %w", err)
	// 	return
	// }

	// return &FC{
	// 	Weight: w,
	// 	Bias:   b,
	// }, nil
}

/* ----- helpers ----- */

func toValidBatchNormConfig(iconf *BatchNormConfig) (conf *BatchNormConfig, err error) {
	if iconf == nil {
		err = fmt.Errorf("expected config not to be nil")
		return
	}

	conf = new(BatchNormConfig)
	*conf = *iconf

	if conf.Dim < 0 {
		err = fmt.Errorf("expected 'Dim' not to be negative: got (%d)", conf.Dim)
		return
	}

	if conf.Momentum < 0 {
		err = fmt.Errorf("expected 'Momentum' not to be negative: got (%f)", conf.Momentum)
		return
	}

	if conf.Eps <= 0 {
		err = fmt.Errorf("expected 'Eps' to be positive: got (%f)", conf.Eps)
		return
	}

	if conf.Initializers == nil {
		conf.Initializers = make(map[string]Initializer)
	}

	if _, ok := conf.Initializers[batchNormBetaKey]; !ok {
		conf.Initializers[batchNormBetaKey] = initializers.NewFull(
			&initializers.FullConfig{
				Value: 0.,
			})
	}

	if _, ok := conf.Initializers[batchNormGammaKey]; !ok {
		conf.Initializers[batchNormGammaKey] = initializers.NewFull(
			&initializers.FullConfig{
				Value: 1.,
			})
	}

	if _, ok := conf.Initializers[batchNormMovingMeanKey]; !ok {
		conf.Initializers[batchNormMovingMeanKey] = initializers.NewFull(
			&initializers.FullConfig{
				Value: 0.,
			})
	}

	if _, ok := conf.Initializers[batchNormMovingVarKey]; !ok {
		conf.Initializers[batchNormMovingVarKey] = initializers.NewFull(
			&initializers.FullConfig{
				Value: 1.,
			})
	}

	if conf.Device == 0 {
		conf.Device = tensor.CPU
	}

	return conf, nil
}

// func validateInitializedWeights(w tensor.Tensor, b tensor.Tensor, conf *FCConfig) (err error) {
// 	shapew := w.Shape()
// 	shapeb := b.Shape()

// 	if len(shapew) != 1 || len(shapeb) != 1 {
// 		err = fmt.Errorf("expected initialized weights to have exactly one dimension")
// 		return
// 	}

// 	if shapew[0] != conf.Outputs {
// 		err = fmt.Errorf("expected initialized 'Weight' size to match 'Outputs': (%d) != (%d)", shapew[0], conf.Outputs)
// 		return
// 	}

// 	if shapeb[0] != conf.Outputs {
// 		err = fmt.Errorf("expected initialized 'Bias' size to match 'Outputs': (%d) != (%d)", shapeb[0], conf.Outputs)
// 		return
// 	}

// 	return nil
// }
