package main

import (
	"bufio"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/stat"
)

type dataSplit struct {
	xTrain [][]float64
	yTrain [][]float64
	xValid [][]float64
	yValid [][]float64
	xTest  [][]float64
	yTest  [][]float64
}

type transformFunc func(float64) float64

func loadData() (x [][]float64, y [][]float64, err error) {
	file, err := os.Open(dataFileAddress)
	if err != nil {
		return
	}

	defer func() {
		err = file.Close()
	}()

	for fscan := bufio.NewScanner(file); fscan.Scan(); {
		line := fscan.Text()
		line = strings.TrimSpace(line)

		if len(line) == 0 {
			break
		}

		fields := strings.Split(line, ",")
		lenf := len(fields)

		xi := make([]float64, lenf-1)
		yi := make([]float64, 0, 3)

		for j := range lenf - 1 {
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

func splitData(x [][]float64, y [][]float64) *dataSplit {
	lend := len(x)
	lenvl := int(float64(lend) * validDataRatio)
	lente := int(float64(lend) * testDataRatio)

	rand.Shuffle(lend, func(i, j int) {
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	})

	point1 := lenvl
	point2 := point1 + lente

	return &dataSplit{
		xValid: x[:point1],
		yValid: y[:point1],
		xTest:  x[point1:point2],
		yTest:  y[point1:point2],
		xTrain: x[point2:],
		yTrain: y[point2:],
	}
}

func preprocessData(data *dataSplit) {
	getColumn := func(j int, x [][]float64) (col []float64) {
		col = make([]float64, len(x))
		for i := range x {
			col[i] = x[i][j]
		}

		return col
	}

	transformColumn := func(j int, x [][]float64, tf transformFunc) {
		for i := range x {
			x[i][j] = tf(x[i][j])
		}
	}

	for j := range data.xTrain[0] {
		col := getColumn(j, data.xTrain)
		stf := makeStandardizer(col)

		transformColumn(j, data.xTrain, stf)
		transformColumn(j, data.xValid, stf)
		transformColumn(j, data.xTest, stf)
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

func makeStandardizer(col []float64) transformFunc {
	u, s := stat.MeanStdDev(col, nil)

	return func(v float64) float64 {
		return (v - u) / s
	}
}
