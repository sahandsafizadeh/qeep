package main

import (
	"bufio"
	"fmt"
	"math/rand/v2"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/component/optimizers"
	"github.com/sahandsafizadeh/qeep/model"
	"github.com/sahandsafizadeh/qeep/model/batchgens"
	"github.com/sahandsafizadeh/qeep/model/stream"
)

const (
	// https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data
	dataFileAddress = "data.csv"
	testDataRatio   = 0.2
)

const (
	batchSize = 32
	epochs    = 50
)

func main() {
	result, err := run()
	if err != nil {
		panic(err)
	}

	fmt.Println(result)
}

func run() (result map[string]float64, err error) {
	trainBatchGen, testBatchGen, err := prepareData()
	if err != nil {
		panic(err)
	}

	bhModel, err := prepareModel()
	if err != nil {
		return
	}

	err = bhModel.Fit(trainBatchGen, &model.FitConfig{Epochs: epochs})
	if err != nil {
		return
	}

	result, err = bhModel.Eval(testBatchGen, nil)
	if err != nil {
		return
	}

	return result, nil
}

/* ----- model preparation ----- */

func prepareModel() (m *model.Model, err error) {
	input := stream.Input()

	x := stream.FC(&layers.FCConfig{
		Inputs:  13,
		Outputs: 64,
	})(input)
	x = stream.Relu()(x)

	x = stream.FC(&layers.FCConfig{
		Inputs:  64,
		Outputs: 32,
	})(x)
	x = stream.Relu()(x)

	output := stream.FC(&layers.FCConfig{
		Inputs:  32,
		Outputs: 1,
	})(x)

	/* -------------------- */

	m, err = model.NewModel(input, output, &model.ModelConfig{
		Loss:      losses.NewMSE(),
		Optimizer: optimizers.NewSGD(nil),
	})
	if err != nil {
		return
	}

	return m, nil
}

/* ----- data preparation ----- */

func prepareData() (trainBatchGen, testBatchGen model.BatchGenerator, err error) {
	x, y, err := loadData()
	if err != nil {
		return
	}

	xtr, xte, ytr, yte := splitData(x, y)

	trainBatchGen, err = batchgens.NewSimple(xtr, ytr, &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   true,
	})
	if err != nil {
		return
	}

	testBatchGen, err = batchgens.NewSimple(xte, yte, &batchgens.SimpleConfig{
		BatchSize: batchSize,
		Shuffle:   true,
	})
	if err != nil {
		return
	}

	return trainBatchGen, testBatchGen, nil
}

func loadData() (x [][]float64, y [][]float64, err error) {
	file, err := os.Open(dataFileAddress)
	if err != nil {
		return
	}

	defer file.Close()

	for fscan := bufio.NewScanner(file); fscan.Scan(); {
		line := fscan.Text()
		line = strings.TrimSpace(line)

		re := regexp.MustCompile(`\s+`)
		fields := re.Split(line, -1)
		lenf := len(fields)

		xi := make([]float64, lenf-1)
		yi := make([]float64, 1)

		for j := 0; j < lenf-1; j++ {
			xi[j] = mustParseFloat64(fields[j])
		}
		yi[0] = mustParseFloat64(fields[lenf-1])

		x = append(x, xi)
		y = append(y, yi)
	}

	return x, y, nil
}

func splitData(x [][]float64, y [][]float64) (xTrain, xTest [][]float64, yTrain, yTest [][]float64) {
	lend := len(x)
	ratio := 1. - testDataRatio
	point := int(float64(lend) * ratio)

	rand.Shuffle(lend, func(i, j int) {
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	})

	xTrain = x[:point]
	yTrain = y[:point]
	xTest = x[point:]
	yTest = y[point:]

	return xTrain, xTest, yTrain, yTest
}

/* ----- helpers ----- */

func mustParseFloat64(s string) (f float64) {
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		panic(err)
	}

	return f
}
