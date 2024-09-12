# Deep Learning Perspective on Systemic Risk with Non-Normal Risk Factors

This project implements deep learning techniques to model **systemic risk**, a key factor in financial stability, particularly during crises. Traditional risk models often rely on normal distributions, which fail to account for the complexities of real-world financial data and extreme events. This thesis addresses these limitations by incorporating non-normal risk factors such as the **Student-t distribution** and **copulas**, which are better suited for capturing tail risks and dependencies. By optimizing cash allocations among financial institutions, the project aims to improve the accuracy of systemic risk predictions and enhance overall financial system stability.

### Key Objectives:
1. **Review and critique existing systemic risk measures** that rely on normally distributed risk factors.
2. **Develop deep learning models** to handle complex, high-dimensional financial data using non-normal distributions, building on methods outlined by Feng et al. (2022) for optimizing systemic risk measures.
3. **Compare the performance** of these models using simulated data to evaluate the effectiveness of deep learning in systemic risk assessment.

## Implementation Details

### Data Simulation:
- **Risk Factors Simulation:** The project simulates financial data using three distributions:
  1. **Normally Distributed Risk Factors**.
  2. **Student-t Distributed Risk Factors** (captures heavy tails).
  3. **Copula Distributed Risk Factors** (models non-linear dependencies).

- Simulated data is used to train the neural networks for systemic risk assessment.

### Neural Networks:
The project uses a combination of **primal and dual problem formulations** based on the work of Feng et al. (2022), solved using neural networks. The deep learning framework TensorFlow is used for model building and training.

- **Primal Problem:** Optimizes cash allocations across financial institutions to ensure the stability of the system.
- **Dual Problem:** Identifies how to distribute systemic risk fairly.

#### Network Architecture:
- **Fully connected neural networks** with 3 hidden layers.
- **Leaky ReLU activation** functions are used in the hidden layers to avoid the dead neurons problem.
- The networks are trained using **Stochastic Gradient Descent (SGD)** with a learning rate and decay.

#### GANs for Systemic Risk:
- The dual problem leverages **Generative Adversarial Networks (GANs)**, where:
  - The **generator** network produces estimates for risk allocations.
  - The **discriminator** network estimates the Radon-Nikodym derivative, a crucial component for adjusting the allocation fairly, as proposed by Feng et al. (2022).


## Installation and usage

### Prerequisites

- Python 3.9.17 and 3.10.14 using `pyenv`.
- TensorFlow and other dependencies specified in the `requirements` files.

### Simulating Risk Factors

The first step is to simulate the risk factors used to train the neural network models. To reproduce the results in the `risk-factors-simulation` notebook, follow these steps:
```bash
git clone https://github.com/vargovema/systemic-risk.git
pyenv local 3.9.17
python -m venv venv39
source venv39/bin/activate 
pip install -r requirements39.txt
```

### Implementing Neural Networks

To implement and train neural networks using the simulated data, follow these steps to reproduce the results in the `sys-risk-norm-dist`, `sys-risk-t-dist`, and `sys-risk-cop-dist` notebooks:
```bash
git clone https://github.com/vargovema/systemic-risk.git
pyenv local 3.10.14
python -m venv venv310
source venv310/bin/activate  
pip install -r requirements310.txt
```

## References
Feng, Y., Min, M., & Fouque, J.-P. (2022). *Deep learning for systemic risk measures.* Retrieved 2024-09-02, from https://arxiv.org/abs/2207.00739 doi: 10.48550/arXiv.2207.00739
