package network

type Synapse struct {
	fromNeuron, toNeuron *Neuron	//Synapse connect 2 neurons
	weight float64	//Synapse has weight
}
