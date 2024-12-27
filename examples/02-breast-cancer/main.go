package main

import (
	"bufio"
	"fmt"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"

	"github.com/sahandsafizadeh/qeep/component/layers"
	"github.com/sahandsafizadeh/qeep/component/losses"
	"github.com/sahandsafizadeh/qeep/component/metrics"
	"github.com/sahandsafizadeh/qeep/component/optimizers"
	"github.com/sahandsafizadeh/qeep/model"
	"github.com/sahandsafizadeh/qeep/model/batchgens"
	"github.com/sahandsafizadeh/qeep/model/stream"
	"gonum.org/v1/gonum/stat"
)

const (
	// https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
	dataFileAddress = "data.csv"
	testDataRatio   = 0.2
)

const (
	batchSize = 32
	epochs    = 10
)

func main() {
	result, err := run()
	if err != nil {
		panic(err)
	}

	for m, r := range result {
		fmt.Printf("%s: %.2f\n", m, r)
	}
}

func run() (result map[string]float64, err error) {
	trainBatchGen, testBatchGen, err := prepareData()
	if err != nil {
		return
	}

	bcmodel, err := prepareModel()
	if err != nil {
		return
	}

	err = bcmodel.Fit(trainBatchGen, &model.FitConfig{
		Epochs: epochs,
	})
	if err != nil {
		return
	}

	result, err = bcmodel.Eval(testBatchGen, map[string]model.Metric{
		"Accuracy": metrics.NewAccuracy(nil),
	})
	if err != nil {
		return
	}

	return result, nil
}

/* ----- model preparation ----- */

func prepareModel() (m *model.Model, err error) {
	input := stream.Input()

	x := stream.FC(&layers.FCConfig{
		Inputs:  30,
		Outputs: 32,
	})(input)
	x = stream.Relu()(x)

	x = stream.FC(&layers.FCConfig{
		Inputs:  32,
		Outputs: 16,
	})(x)
	x = stream.Relu()(x)

	x = stream.FC(&layers.FCConfig{
		Inputs:  16,
		Outputs: 1,
	})(x)
	output := stream.Sigmoid()(x)

	/* -------------------- */

	m, err = model.NewModel(input, output, &model.ModelConfig{
		Loss:      losses.NewBCE(),
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

	preprocessData(xtr, xte)

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

		fields := strings.Split(line, ",")
		lenf := len(fields)

		xi := make([]float64, lenf-2)
		yi := make([]float64, 1)

		for j := 0; j < lenf-2; j++ {
			xi[j] = mustParseFloat64(fields[j+2])
		}

		switch fields[1] {
		case "M":
			yi[0] = 1.
		case "B":
			yi[0] = 0.
		default:
			panic("unreachable")
		}

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

func preprocessData(xTrain, xTest [][]float64) {
	getColumn := func(j int, x [][]float64) (col []float64) {
		col = make([]float64, len(x))
		for i := 0; i < len(x); i++ {
			col[i] = x[i][j]
		}

		return col
	}

	setColumn := func(j int, x [][]float64, col []float64) {
		for i := 0; i < len(x); i++ {
			x[i][j] = col[i]
		}
	}

	for j := 0; j < len(xTrain[0]); j++ {
		xtrc := getColumn(j, xTrain)
		xtec := getColumn(j, xTest)
		standardize(xtrc, xtec)
		setColumn(j, xTrain, xtrc)
		setColumn(j, xTest, xtec)
	}
}

/* ----- helpers ----- */

func mustParseFloat64(s string) (f float64) {
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		panic(err)
	}

	return f
}

func standardize(xtrc, xtec []float64) {
	u, s := stat.MeanStdDev(xtrc, nil)
	transform := func(v float64) float64 { return (v - u) / s }

	transformColumn := func(col []float64) {
		for i := 0; i < len(col); i++ {
			col[i] = transform(col[i])
		}
	}

	transformColumn(xtrc)
	transformColumn(xtec)
}
