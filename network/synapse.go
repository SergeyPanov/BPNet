package network

type Synapse struct {
	fromNeuron, toNeuron *Neuron	//Synapse connect 2 neurons
	weight, deltaPrevWeight float64	//Synapse has weight
}
