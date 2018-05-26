package network

import (
	"math/rand"
	"fmt"
)

type Network struct {
	input, hidden, output []Neuron
}


//Create input, hidden, output layers of neurons and connect them

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
			synapse := Synapse{&net.hidden[i], &net.output[j], rand.Float64()}
			net.hidden[i].oSynapse = append(net.hidden[i].oSynapse, &synapse)
			net.output[j].iSynapse = append(net.output[j].iSynapse, &synapse)
		}
	}
}

//Calculate output
func (net *Network) Calculate(vec []float64) []float64 {

	outputVector := make([]float64, len(net.output))

	//Fire of input neuron is an input vector
	for i, _ := range net.input{
		net.input[i].fire = vec[i]
	}

	//Calculate fires of hidden layer
	for i, hn := range net.hidden{
		inpValue := 0.0
		for _, is := range hn.iSynapse{
			inpValue += is.fromNeuron.fire * is.weight
		}
		net.hidden[i].fire = hn.Sigmoid(inpValue)
	}

	//Calculate fire of output neuron
	for i, on := range net.output{
		inpValue := 0.0
		for _, is := range on.iSynapse{
			inpValue += is.fromNeuron.fire * is.weight
		}
		net.output[i].fire = on.Sigmoid(inpValue)

		outputVector[i] = on.fire
	}
	return outputVector

}

//Delta for output layer
func (net *Network) deltaOut(ideal []float64) []float64 {
	deltaO := make([]float64, len(net.output))
	for i := 0; i < len(net.output); i++  {
		deltaO[i] = (ideal[i] - net.output[i].fire) * net.output[i].DSigmoid()
		net.output[i].delta = (ideal[i] - net.output[i].fire) * net.output[i].DSigmoid()
	}
	return deltaO
}
//Delta for hidden layer
func (net *Network) deltaHidden(index int) {
	sum := 0.0
	for i := 0; i < len(net.hidden[index].oSynapse) ; i++ {
		sum += net.hidden[index].oSynapse[i].weight * net.hidden[index].oSynapse[i].toNeuron.delta
	}
	net.hidden[index].delta = sum * net.hidden[index].DSigmoid()

	fmt.Println("DELTA HIDDEN: ")
	fmt.Println(net.hidden[index].delta )

}

func (net *Network) Learn(ideal []float64) {

	deltaO := net.deltaOut(ideal)

	net.deltaHidden(0)


	fmt.Println("DELTA O: ")
	fmt.Println(deltaO)
	fmt.Println(net.output)
}



func (net *Network) Dump(){
	fmt.Println("Input layer: ")
	for i, in := range net.input{
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