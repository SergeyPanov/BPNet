package main

import (
	"BPNet/network"
)

func main() {
	net := new(network.Network)
	net.Init(2, 2, 1)
	net.Dump()
}
