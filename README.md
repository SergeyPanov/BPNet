# BPNet
Simple neural network using the [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) method.

# Compilation
Command `make` creates an executable binary file `bp`. Another way is custom compilation using `go build`: `go build -o bp`.

# Run
Program is ran as follow:
```
Usage of ./bp:
  -dv float
        Stop condition is deviation. (default -1)
  -ep int
        Stop condition is number of epochs. (default -1)
  -hl int
        Hidden layer.
  -is string
        Path to ideal set. (default "./resources/wine-ideal")
  -lr float
        Learning rate. (default 0.7)
  -m float
        Momentum. (default 0.3)
  -ts string
        Path to training set. (default "./resources/wine-input")
  -us string
        Path to unknown set. (default "./resources/wine-unknown")
```

# Examples
1. `./bp -hl=6 -m=0.5 -dv=0.03`: 6 hidden neurons, 0.5 is moment parameter, 0.03 is an error deviation.
2. `./bp -ep=1000 -hl=19`: 19 hidden neurons, 1000 epochs of learning

# Output format description
`[vals] error`: `[vals]` is a vector of values, `error` is an [Root MSE](https://en.wikipedia.org/wiki/Mean_squared_error).
