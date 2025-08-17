# Basic-Neural-Network

This repo is a self-contained (no external libraries), minimal neural network implementation in C++.

Exploration of the [German tank problem](https://en.wikipedia.org/wiki/German_tank_problem) is used as the motivational demonstration.

---

##  Overview

Its primary purpose is self-directed learning to understand the fundamentals of neural network–based machine learning. It includes:

- Fully connected layers with customizable architecture
- Forward propagation and backpropagation using the sigmoid activation function
- Training via stochastic gradient descent (SGD)
- No external dependencies — just standard C++ STL and `<cmath>`

The MNIST data-loading logic is inspired by and adapted from [Krish120003's C++ implementation](https://github.com/Krish120003/CPP_Neural_Network).

---

##  Implementation Principles

The following guiding principles governed the implementation decisions of the project:
- **Prefer clear implementation over efficiency and performance**. Classes and methods should have clear analogues to intuitive neural network concepts. Matrix maths has been avoided in favour of explicit propagation logic.
- **Be lightweight and self-contained**. Supporting the first point, avoid pre-built implementations to focus on fundamentals and clear implementation.
- **Focus on maintainability and extensibility**. Designed to be a foundation project upon which to explore further concepts (e.g., ReLU, batch training, alternative loss functions).

---

##  Getting Started

### Prerequisites

- A modern C++ compiler (e.g. `g++`, `clang++`) supporting C++11 or later.
- `make` to build the project (or compile manually).
- *NOTE: Only tested on Linux (Debian-based) distributions so far*

### Building

```bash
git clone https://github.com/jpmoranpac/Basic-Neural-Network.git
cd Basic-Neural-Network
make
```

This will compile the source into the executable `neural_network.out`.

Usage
`./bin/neural_network.out`

*NOTE: There is no command-line parsing yet, modify constants directly in demo implementation.*

## Project Structure

```
Basic-Neural-Network/
├── src/        # Source code — Neuron, Layer,
                # Network classes + training logic
├── data/       # Example data (e.g. MNIST 
                # formatted files)
├── makefile    # Build instructions
└── README.md   # This documentation
```

## Extending the Network

Enhancements planned for this project:

- Modify the German tank problem to be more favourable to a neural network approach. For example, a sampling bias, truncating the population so only the first *N* tanks are observed, observation noise in the serial numbers, sampling with replacement, etc.
- Swap the sigmoid activation for alternatives like ReLU, tanh, or identity (for regression).
- Support mini-batch SGD or momentum-based optimizers.
- Add different loss functions (e.g., cross-entropy).
- Introduce regularization (L1/L2 dropout).
- Allow selecting activation/loss functions at runtime or compile time.
- Add logging of loss/accuracy for visualization.

## Acknowledge and Thanks

Inspired by Krish120003’s C++ neural network.