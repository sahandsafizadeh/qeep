# qeep

Welcome to **qeep** (pronounced /kēp/)! This project implements a **_deep learning framework_** in pure **_Go_**. It allows you to define neural networks in a declarative way while being able to control operations at _tensor level_.

## Features

- _Multi-Dimensional_ **Tensors** with a variety of linear algebra and statistical operations. (`GPU` acceleration is planned for future development)
- Automatic differentiation (_AutoGrad_) for tensors.
- Variety of neural network _Components_ such as `FC` (fully-connected).
- A _Declarative_ approach to define neural networks using `stream` package.

## Installation

Ensure that you have [Go](https://go.dev/dl/) installed. Then, you can install the package using _Go modules_:

```bash
go get github.com/sahandsafizadeh/qeep
go mod tidy
```

## Usage

Here is an example of defining a classification model:

```go
package main

import (
	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/layers/activations"
	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/component/optimizers"
	"github.com/sahandsafizadeh/qeep/model"
	"github.com/sahandsafizadeh/qeep/model/stream"
)

func prepareModel() (m *model.Model, err error) {
	input := stream.Input()

	x := stream.FC(&layers.FCConfig{
		Inputs:  4,
		Outputs: 4,
	})(input)
	x = stream.Tanh()(x)

	x = stream.FC(&layers.FCConfig{
		Inputs:  4,
		Outputs: 3,
	})(x)
	output := stream.Softmax(&activations.SoftmaxConfig{Dim: 1})(x)

	/* ---------- */

	m, err = model.NewModel(input, output, &model.ModelConfig{
		Loss:      losses.NewCE(),
		Optimizer: optimizers.NewSGD(&optimizers.SGDConfig{LearningRate: 1e-5}),
	})
	if err != nil {
		return
	}

	return m, nil
}
```

More working [examples](./examples) are provided. You can download their dataset and run them like the following for _Iris Classification_:

```bash
cd ./examples/03-Iris/
curl -o data.csv https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
go run .
```

## License

The **qeep** project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
