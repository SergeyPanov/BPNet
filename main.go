package main

import (
	"BPNet/network"
	"fmt"
)

func main() {
	net := new(network.Network)
	net.Init(2, 2, 1, 0.7, 0.3)

	input := [][]float64{{1.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}}
	ideal := [][]float64{{1.0}, {1.0}, {1.0}, {0.0}}

	net.LearnEpochs(10000, input, ideal)

	net.LearnError(0.01, input, ideal)
	net.Calculate([]float64{0.0, 0.0})
	fmt.Println(net.Result())

}
