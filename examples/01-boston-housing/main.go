package main

import (
	"encoding/csv"
	"fmt"
	"os"
)

func main() {
	const dataFile = "data.csv"

	file, err := os.Open(dataFile)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = ' '

	for {
		record, err := reader.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			panic(err)
		}
		fmt.Println(record)
	}
}
