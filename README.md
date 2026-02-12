# SafeSplit: Defending Against Model Poisoning in Split Learning

This repository contains the implementation of **SafeSplit**, a defense mechanism against client-side model poisoning attacks in Split Learning. It supports simulation on MNIST and CIFAR-10 datasets with both IID and Non-IID data distributions.

## Features
-   **Split Learning Simulation**: Modular client-server architecture simulation.
-   **Defenses**: SafeSplit (spectral filtering) defense implemented.
-   **Datasets**: MNIST, CIFAR-10.
-   **Data Distribution**: IID and Non-IID (Dirichlet sampling).
-   **Experimentation**: Notebooks for single-run and multi-seed statistical analysis.

## Structure
-   `models.py`: Neural network architectures (Head, Tail, Backbone).
-   `safesplit.py`: Core SafeSplit defense implementation (DCT/IDCT, filtering).
-   `simulation.py`: Client and Server classes, main simulation loop.
-   `data.py`: Dataset loading, poisoning, and partitioning (IID/Non-IID).
-   `experiment.ipynb`: Main notebook for configuring and running simulations.
-   `multi_seed_experiment.ipynb`: Comparison notebook running multiple seeds with confidence intervals.

## Dependencies
-   Python 3.8+
-   PyTorch
-   Torchvision
-   Numpy
-   Matplotlib
-   Scipy

Install via:
```bash
pip install -r requirements.txt
```

## Usage
1.  Open `experiment.ipynb` or `multi_seed_experiment.ipynb`.
2.  Configure parameters (dataset, distribution, alpha, defense, etc.).
3.  Run the cells.

## Remote Execution (Colab/Kaggle)
The notebooks include a setup cell to clone this repository and install dependencies automatically. Ensure you provide the correct GitHub repository URL in the first cell if forked.
