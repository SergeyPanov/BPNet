package network

import (
	"fmt"
	"math/rand"
)

type Network struct {
	input, hidden, output []Neuron
}

/*
	Create input, hidden, output layers of neurons and connect them
 */
func (net *Network) Init(in, hd, ou int)  {
	net.input = make([]Neuron, in)
	net.hidden = make([]Neuron, hd)
	net.output = make([]Neuron, ou)

	//Connect input layer and hidden layer
	for i := 0; i < in; i++ {
		for j := 0; j < hd; j++ {
			synapse := Synapse{&net.input[i], &net.hidden[j], rand.Float64()}
			net.input[i].oSynapse = append(net.input[i].oSynapse, &synapse)
			net.hidden[j].iSynapse = append(net.hidden[j].iSynapse, &synapse)
		}
	}

	//Connect hidden layer and output layer
	for i := 0; i < hd; i++ {
		for j := 0; j < ou; j ++ {
			synapse := Synapse{&net.hidden[i], &net.output[j], rand.ExpFloat64()}
			net.hidden[i].oSynapse = append(net.hidden[i].oSynapse, &synapse)
			net.output[j].iSynapse = append(net.output[j].iSynapse, &synapse)
		}
	}
}

//Calculate output
func (net *Network) Calculate(vec []float64) []float64 {

	outputVector := make([]float64, len(vec))

	//Fire of input neuron is an input vector
	for i, in := range net.input{
		in.fire = vec[i]
	}

	//Calculate fires of hidden layer
	for _, hn := range net.hidden{
		inpValue := 0.0
		for _, is := range hn.iSynapse{
			inpValue += is.fromNeuron.fire * is.weight
		}
		hn.fire = hn.Sigmoid(inpValue)
	}

	//Calculate fire of output neuron
	for i, on := range net.output{
		inpValue := 0.0
		for _, is := range on.iSynapse{
			inpValue += is.fromNeuron.fire * is.weight
		}
		on.fire = on.Sigmoid(inpValue)

		outputVector[i] = on.fire
	}
	return outputVector

}

func (net *Network) Dump(){
	fmt.Println("Input layer: ")
	for i, in := range net.hidden{
		fmt.Println("Fire of nuron: ", i, " is: ", in.fire)
		fmt.Println("Output synapses:")
		for _, os := range in.oSynapse{
			fmt.Println("Weight: ", os.weight)
		}
	}

	fmt.Println("----------------------------")
	fmt.Println("Hidden layer: ")

	for i, hn := range net.hidden{
		fmt.Println("Fire of nuron: ", i, " is: ", hn.fire)
		fmt.Println("Output synapses:")
		for _, os := range hn.oSynapse{
			fmt.Println("Weight: ", os.weight)
		}
	}

	fmt.Println("----------------------------")
	fmt.Println("Output layer: ")
	for i, on := range net.output{
		fmt.Println("Fire of nuron: ", i, " is: ", on.fire)
	}


}