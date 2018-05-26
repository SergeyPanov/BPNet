package main

import (
	"BPNet/network"
)

func main() {
	net := new(network.Network)
	net.Init(2, 2, 1)
	net.Calculate([]float64{1.0, 0.0})
	net.Dump()
	net.Learn([]float64{1})
}
