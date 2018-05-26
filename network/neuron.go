package network

import "math"

type Neuron struct {
	fire float64
	iSynapse, oSynapse []*Synapse	//Neuron has input/output synapses
}

// Activation function "sigmoid"
func (n *Neuron) Sigmoid(x float64) float64  {
	return 1.0 / (1 + math.Pow( math.E, (-x) ) )
}