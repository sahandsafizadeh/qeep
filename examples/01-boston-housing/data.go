package main

import (
	"bufio"
	"math/rand/v2"
	"os"
	"regexp"
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
		for i := 0; i < len(x); i++ {
			col[i] = x[i][j]
		}

		return col
	}

	transformColumn := func(j int, x [][]float64, tf transformFunc) {
		for i := 0; i < len(x); i++ {
			x[i][j] = tf(x[i][j])
		}
	}

	for j := 0; j < len(data.xTrain[0]); j++ {
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
