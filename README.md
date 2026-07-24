# qeep

Welcome to **qeep** (pronounced /kēp/)! This project implements a **deep learning framework** in **_Go_**. It allows you to define neural networks in a declarative way while being able to control operations at tensor level.

## Features

- 📐 _Multi-Dimensional_ **Tensors** with a wide range of linear algebra and statistical operations.
- 🔁 _Automatic differentiation_ (**AutoGrad**) for tensors.
- ⚡ _GPU acceleration_ via **CUDA** for high-performance computations.
- 🧱 A variety of neural network **components**, such as `FC` (fully connected), `BatchNorm`, and `Dropout` layers.
- 📝 A _declarative API_ for defining neural networks using `stream` package.

## Installation

Navigate to your project directory and clone the qeep repository:

```bash
git clone https://github.com/sahandsafizadeh/qeep
```

Ensure that you have [Go](https://go.dev/dl/) installed. Then, Link the local qeep package to your project using _Go modules_:

```go
go mod edit -require=github.com/sahandsafizadeh/qeep@v0.6.0
go mod edit -replace=github.com/sahandsafizadeh/qeep=./qeep
go mod tidy
```

💡 This setup assumes that qeep is cloned into a local folder named `./qeep` relative to your Go project.

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
	"github.com/sahandsafizadeh/qeep/tensor"
)

const dev = tensor.CPU

func prepareModel() (m *model.Model, err error) {
	input := stream.Input()

	x := stream.FC(&layers.FCConfig{Outputs: 256, Device: dev})(input)
	x = stream.BatchNorm(&layers.BatchNormConfig{Device: dev})(x)
	x = stream.Relu()(x)
	x = stream.Dropout(&layers.DropoutConfig{Rate: 0.2})(x)

	x = stream.FC(&layers.FCConfig{Outputs: 128, Device: dev})(x)
	x = stream.BatchNorm(&layers.BatchNormConfig{Device: dev})(x)
	x = stream.Relu()(x)
	x = stream.Dropout(&layers.DropoutConfig{Rate: 0.2})(x)

	x = stream.FC(&layers.FCConfig{Outputs: 10, Device: dev})(x)
	output := stream.Softmax(&activations.SoftmaxConfig{Dim: 1})(x)

	/* -------------------- */

	loss := losses.NewCE()

	optimizer, err := optimizers.NewAdam(nil)
	if err != nil {
		return m, err
	}

	m, err = model.NewModel(input, output, &model.ModelConfig{
		Loss:      loss,
		Optimizer: optimizer,
	})
	if err != nil {
		return m, err
	}

	return m, nil
}
```

📂 More working [examples](./examples) are provided. You can **download their dataset** and run them like the following for _MNIST_ Classification:

```bash
cd ./qeep/examples/04-mnist/
go run .
```

## Running on GPU

### Prerequisites

1. An accessible _CUDA-capable GPU_.
2. [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed.
3. A working _C Toolchain_ like `gcc`.
4. _CGO_ enabled and configured ([setup guide](https://github.com/go101/go101/wiki/CGO-Environment-Setup)).

### Run with CUDA

Build the necessary CUDA libraries once:

```bash
cd ./qeep
make cuda
```

Now in your Go code, you can set devices to `tensor.CUDA`.

🔥 Finally, run your program with the `cuda` build tag:

```bash
go run -tags=cuda .
```

## License

The **qeep** project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
