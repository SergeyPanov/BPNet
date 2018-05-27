package network

import (
	"math/rand"
	"time"
	"math"
)

type Network struct {
	input, hidden, output []Neuron
	e, m, biasI, biasH    float64
	avgRootMSE            float64
}

//Create input, hidden, output layers of neurons and connect them
func (net *Network) Init(in, hd, ou int, e, m float64) {
	rand.Seed(time.Now().UTC().UnixNano())
	net.e = e
	net.m = m
	net.avgRootMSE = 1
	net.biasI = rand.Float64()
	net.biasH = rand.Float64()
	net.input = make([]Neuron, in)
	net.hidden = make([]Neuron, hd)
	net.output = make([]Neuron, ou)

	//Connect input layer and hidden layer
	for i := 0; i < in; i++ {
		for j := 0; j < hd; j++ {
			synapse := Synapse{&net.input[i], &net.hidden[j], rand.Float64(), 0.0}
			net.input[i].oSynapse = append(net.input[i].oSynapse, &synapse)
			net.hidden[j].iSynapse = append(net.hidden[j].iSynapse, &synapse)
		}
	}

	//Connect hidden layer and output layer
	for i := 0; i < hd; i++ {
		for j := 0; j < ou; j ++ {
			synapse := Synapse{&net.hidden[i], &net.output[j], rand.Float64(), 0.0}
			net.hidden[i].oSynapse = append(net.hidden[i].oSynapse, &synapse)
			net.output[j].iSynapse = append(net.output[j].iSynapse, &synapse)
		}
	}
}

//Calculate output
func (net *Network) Calculate(vec []float64) []float64 {

	outputVector := make([]float64, len(net.output))

	//Fire of input neuron is an input vector
	for i := range net.input {
		net.input[i].fire = vec[i]
	}

	//Calculate fires of hidden layer
	for i, hn := range net.hidden {
		inpValue := net.biasI
		for _, is := range hn.iSynapse {
			inpValue += is.fromNeuron.fire * is.weight
		}
		net.hidden[i].fire = hn.Sigmoid(inpValue)
	}

	//Calculate fire of output neuron
	for i, on := range net.output {
		inpValue := net.biasH
		for _, is := range on.iSynapse {
			inpValue += is.fromNeuron.fire * is.weight
		}
		net.output[i].fire = on.Sigmoid(inpValue)

		outputVector[i] = on.fire
	}
	return outputVector
}

//Delta for output layer
func (net *Network) deltaOut(ideal []float64) {
	for i := 0; i < len(net.output); i++ {
		net.output[i].delta = (ideal[i] - net.output[i].fire) * net.output[i].DSigmoid()
	}
}

//Delta for hidden layer
func (net *Network) deltaHidden(index int) {
	sum := 0.0
	for i := 0; i < len(net.hidden[index].oSynapse); i++ {
		sum += net.hidden[index].oSynapse[i].weight * net.hidden[index].oSynapse[i].toNeuron.delta
	}
	net.hidden[index].delta = sum * net.hidden[index].DSigmoid()
}

//Calculate gradient and delta, update weights base on calculated values
func (net *Network) updateWeights(neuron *Neuron) {
	for i := 0; i < len(neuron.oSynapse); i++ {
		grad := neuron.oSynapse[i].toNeuron.delta * neuron.fire
		deltaW := net.e*grad + net.m*neuron.oSynapse[i].deltaPrevWeight
		neuron.oSynapse[i].deltaPrevWeight = deltaW
		neuron.oSynapse[i].weight += deltaW
	}
}

// Execute one step of learning process
func (net *Network) learningStep(ideal []float64) {
	net.deltaOut(ideal)
	for i := 0; i < len(net.hidden); i++ {
		net.deltaHidden(i)
		net.updateWeights(&net.hidden[i])
	}

	for i := 0; i < len(net.input); i++ {
		net.updateWeights(&net.input[i])
	}
}

// Calculate root MSE
func (net *Network) calculateRootMSE(ideal []float64) float64 {
	divided := 0.0
	for i := range ideal {
		divided += math.Pow(ideal[i]-net.output[i].fire, 2)
	}
	return math.Sqrt(divided / (float64(len(ideal))))
}

// Execute learning process. Stop condition is number of epochs
func (net *Network) LearnEpochs(epoch int, input [][]float64, ideals [][]float64) {

	for e := 0; e < epoch; e++ {
		totalError := 0.0
		for i := 0; i < len(input); i++ {
			net.Calculate(input[i])
			totalError += net.calculateRootMSE(ideals[i])
			net.learningStep(ideals[i])
		}
		net.avgRootMSE = totalError / float64(len(input))
	}
}

// Execute learning process. Stop condition is deviation
func (net *Network) LearnError(deviation float64, input [][]float64, ideals [][]float64) {

	for net.avgRootMSE > deviation {
		totalError := 0.0
		for i := 0; i < len(input); i++ {
			net.Calculate(input[i])
			totalError += net.calculateRootMSE(ideals[i])
			net.learningStep(ideals[i])
		}
		net.avgRootMSE = totalError / float64(len(input))
	}
}

// Return result
func (net *Network) Result() ([]float64, float64) {
	res := make([]float64, len(net.output))
	for i := 0; i < len(net.output); i++ {
		res[i] = net.output[i].fire
	}
	return res, net.avgRootMSE
}
