package main

import "C"

import "gonum.org/v1/gonum/stat/distuv"

/* ----- random generators ----- */

//export goUniformRand
func goUniformRand(l C.double, u C.double) (r C.double) {
	lower := float64(l)
	upper := float64(u)
	randgen := distuv.Uniform{Min: lower, Max: upper}
	return (C.double)(randgen.Rand())
}

//export goNormalRand
func goNormalRand(u C.double, s C.double) (r C.double) {
	mean := float64(u)
	sigm := float64(s)
	randgen := distuv.Normal{Mu: mean, Sigma: sigm}
	return (C.double)(randgen.Rand())
}

/* ----------------------------- */

func main() {}
