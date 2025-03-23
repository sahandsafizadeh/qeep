package main

import (
	/*
	   #cgo LDFLAGS: -L. -lcudahello
	   extern void cuda_hello();
	*/
	"C"
	"fmt"
)

func main() {
	fmt.Println("Hello from Host!")
	C.cuda_hello()
}
