package main

import (
	"bufio"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"
)

type dataSplit struct {
	xTrain [][]float64
	yTrain [][]float64
	xValid [][]float64
	yValid [][]float64
	xTest  [][]float64
	yTest  [][]float64
}

func prepareData() (data *dataSplit, err error) {
	xTrain, yTrain, err := loadCSV(trainFileAddress)
	if err != nil {
		return data, err
	}

	xTest, yTest, err := loadCSV(testFileAddress)
	if err != nil {
		return data, err
	}

	lend := len(xTrain)
	lenvl := int(float64(lend) * validDataRatio)

	rand.Shuffle(lend, func(i, j int) {
		xTrain[i], xTrain[j] = xTrain[j], xTrain[i]
		yTrain[i], yTrain[j] = yTrain[j], yTrain[i]
	})

	return &dataSplit{
		xValid: xTrain[:lenvl],
		yValid: yTrain[:lenvl],
		xTrain: xTrain[lenvl:],
		yTrain: yTrain[lenvl:],
		xTest:  xTest,
		yTest:  yTest,
	}, nil
}

func preprocessData(data *dataSplit) {
	scale := func(rows [][]float64) {
		for i := range rows {
			for j := range rows[i] {
				rows[i][j] /= 255.0
			}
		}
	}

	scale(data.xTrain)
	scale(data.xValid)
	scale(data.xTest)
}

/* ----- helpers ----- */

func loadCSV(fileAddress string) (x [][]float64, y [][]float64, err error) {
	file, err := os.Open(fileAddress)
	if err != nil {
		return x, y, err
	}

	defer func() {
		err = file.Close()
	}()

	heading := true
	for fscan := bufio.NewScanner(file); fscan.Scan(); {
		if heading {
			heading = false
			continue
		}

		line := fscan.Text()
		line = strings.TrimSpace(line)

		if len(line) == 0 {
			break
		}

		fields := strings.Split(line, ",")
		lenf := len(fields)

		xi := make([]float64, lenf-1)
		for j := range lenf - 1 {
			xi[j] = mustParseFloat64(fields[j+1])
		}

		yi := labelToOneHot(fields[0])

		x = append(x, xi)
		y = append(y, yi)
	}

	return x, y, nil
}

func labelToOneHot(label string) []float64 {
	switch label {
	case "0":
		return []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	case "1":
		return []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0}
	case "2":
		return []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}
	case "3":
		return []float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0}
	case "4":
		return []float64{0, 0, 0, 0, 1, 0, 0, 0, 0, 0}
	case "5":
		return []float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0}
	case "6":
		return []float64{0, 0, 0, 0, 0, 0, 1, 0, 0, 0}
	case "7":
		return []float64{0, 0, 0, 0, 0, 0, 0, 1, 0, 0}
	case "8":
		return []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0}
	case "9":
		return []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
	default:
		panic("unreachable")
	}
}

func mustParseFloat64(s string) float64 {
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		panic(err)
	}

	return f
}
