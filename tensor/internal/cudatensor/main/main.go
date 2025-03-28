package main

import (
	"fmt"
	"runtime"
	"time"

	"github.com/sahandsafizadeh/qeep/tensor/internal/cudatensor"
)

func main() {
	for range 1000 {
		_, err := cudatensor.Full([]int{256, 128, 64}, 0., false)
		if err != nil {
			panic(err)
		}
	}

	runtime.GC()

	fmt.Println("End of Main")
	time.Sleep(30 * time.Second)
}
