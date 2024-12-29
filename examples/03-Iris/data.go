package main

import (
	"bufio"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/stat"
)

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

		xi := make([]float64, lenf-1)
		yi := make([]float64, 0, 3)

		for j := 0; j < lenf-1; j++ {
			xi[j] = mustParseFloat64(fields[j])
		}

		switch fields[lenf-1] {
		case "Iris-setosa":
			yi = append(yi, 1., 0., 0.)
		case "Iris-versicolor":
			yi = append(yi, 0., 1., 0.)
		case "Iris-virginica":
			yi = append(yi, 0., 0., 1.)
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
