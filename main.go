package main

import (
	"BPNet/network"
	"flag"
	"os"
	"io/ioutil"
	"strings"
	"strconv"
	"fmt"
)

func readContent(path string)[][]float64  {
	bs, err := ioutil.ReadFile(path)
	if err != nil {
		print("File ", path, " does not exists.")
		os.Exit(1)
	}
	var inputs [][]float64
	strs := strings.Split(string(bs), "\n")
	for i := 0; i < len(strs); i++ {
		numbers := strings.Split(strs[i], " ")

		var convertedNumbers []float64

		for n := 0; n < len(numbers); n++{
			converted, err := strconv.ParseFloat(numbers[n], 64)
			if err != nil {
				print("Invalid file")
				os.Exit(1)
			}
			convertedNumbers = append(convertedNumbers, converted)
		}
		inputs = append(inputs, convertedNumbers)
	}

	return inputs

}

func main() {
	net := new(network.Network)

	learningRate := flag.Float64("lr", 0.7, "Learning rate")
	momentum := flag.Float64("m", 0.3, "Momentum")
	inputLayer := flag.Int("il", 0, "Input layer")
	hiddenLayer := flag.Int("hl", 0, "Hidden layer")
	outputLayer := flag.Int("ol", 0, "Output layer")

	trainingSet := flag.String("ts", "./resources/wine-input", "Path to training set")
	idealSet := flag.String("is", "./resources/wine-ideal", "Path to ideal set")
	unknownSet := flag.String("us", "./resources/wine-unknown", "Path to unknown set")

	flag.Parse()
	readContent(*trainingSet)

	ts := readContent(*trainingSet)
	is := readContent(*idealSet)
	us := readContent(*unknownSet)

	net.Init(*inputLayer, *hiddenLayer, *outputLayer, *learningRate, *momentum)


	net.LearnEpochs(100, ts, is)

	for i := 0; i < len(us);  i++ {
		net.Calculate(us[i])
		fmt.Println(net.Result())
	}



}
