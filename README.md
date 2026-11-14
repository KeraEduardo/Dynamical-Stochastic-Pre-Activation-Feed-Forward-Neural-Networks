# Dynamical Stochastic Pre-Activation Feed-Forward Neural Networks

In this repository you will find all code used to numerically-simulate dynamical stochastic pre-activation (DSP) feed-forward neural networks (FFN). This code is partitioned into three parts:
1. The FPE folder contains the code used to numerically-simulate the Fokker-Planck Equation (FPE) for a DSP FFN. Although the code written works for any DSP FFN size, I only numerically simulated the case of a single DSP neuron rather than a network as the simulated pdf becomes intractable.
2. The Deterministic folder contains the code to numerically-simulate a DSP FFN in the deterministic domain. This folder contains the code to train a given neural network architecture using backpropagation and three different visualisation files. Visualisation and simulation are treated as separate processes in order to facilitate statistical analyses and visualisation tasks without having to simulate everything all at once. To simulate a DSP FFN in the deterministic domain, the same code as the stochastic domain, which is now explained.
3. The Stochastic folder contains the code to numerically-simulate a DSP FFN in the stochastic domain. This folder contains the code to train a given neural network architecture using Bayes-by-Backprop, a visualisation file, a file to sample from stochastic ANNs and simulate the SDE for a DSP FFN.


A few points:
1. The same code to simulate a DSP FFN is used in both the stochastic and deterministic domains, as deterministic evolution is a special case of stochastic evolution. To obtain deterministic evolution, the thermal fluctuations (or, equivalently) the variances of the biases are all set to zero.
2. Similarly, the same code for sampling was used in both the stochastic and deterministic domains. In this case, a deterministic model is built rather than a stochastic one and only one solution is obtained per datum, unlike the stochastic case where instances are obtained per datum.
3. Most files require a .txt file called "setup.txt", which can be set by the user and contains information such as the architecture used, the dataset, batch_size, if training in deterministic or stochastic domains and hyperparameter values in SI units for the DSP FFN.
