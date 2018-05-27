package network

import "math"

type Neuron struct {
	fire, delta        float64
	iSynapse, oSynapse []*Synapse //Neuron has input/output synapses
}

// Activation function "sigmoid"
func (n *Neuron) Sigmoid(x float64) float64 {
	return 1.0 / (1 + math.Pow(math.E, (-x)))
}

// Derivation of sigmoid
func (n *Neuron) DSigmoid() float64 {
	return (1 - n.fire) * n.fire
}
