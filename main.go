package main

import (
	"BPNet/network"
	"fmt"
)

func main() {
	net := new(network.Network)
	net.Init(2, 2, 1, 0.7, 0.3)

	for i := 0; i < 100 ; i ++  {

		net.Calculate([]float64{1.0, 0.0})
		net.Learn([]float64{0.0})
	}

	fmt.Println(net.Result())


}
