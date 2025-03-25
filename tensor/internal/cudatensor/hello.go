package main

/*
   #cgo LDFLAGS: -L. -lcudahello
   extern void cuda_hello();
*/
import "C"

import "fmt"

func main() {
	fmt.Println("Hello from Host!")
	C.cuda_hello()
}
