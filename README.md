# Time Series Forecasting using LSTM, GRU, RNN, and TCN

This project implements and compares four neural network architectures for small-scale time series forecasting:
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Recurrent Neural Network (RNN)
- Temporal Convolutional Network (TCN)

The goal is to forecast continuous time-dependent values using synthetic tabular data. The dataset represents a single time series with random variations in amplitude, frequency, and noise to simulate realistic temporal behavior.

---

## Project Overview

Time series forecasting aims to predict future values based on previously observed data.  
In this project, we focus on a small-scale setup—short time series with limited features—to demonstrate the behavior and performance of different sequence modeling architectures.

The workflow includes:
1. Generating synthetic time series data.
2. Preparing sequences for supervised learning.
3. Training the LSTM, GRU, RNN, and TCN models.
4. Evaluating model performance on test data.
5. Visualizing actual vs. predicted results.

---

## Model Descriptions

### 1. Long Short-Term Memory (LSTM)
LSTM networks are a variant of RNNs designed to overcome the vanishing gradient problem.  
They maintain long-term dependencies using cell states and gating mechanisms:
- **Input gate:** Decides which values to update.
- **Forget gate:** Determines which information to discard.
- **Output gate:** Controls the output from the cell state.

LSTMs are well-suited for sequential data and can model long-range dependencies effectively.

### 2. Gated Recurrent Unit (GRU)
GRUs simplify the LSTM architecture by combining the input and forget gates into a single update gate while retaining the reset gate. They offer competitive performance on many sequence tasks with fewer parameters, making them faster to train and less prone to overfitting when data is limited.

### 3. Recurrent Neural Network (RNN)
A basic RNN maintains a hidden state that captures information from previous time steps.  
However, standard RNNs often struggle with long-term dependencies due to gradient vanishing during backpropagation.  
They remain useful for shorter sequences and as a baseline for sequence modeling tasks.

### 4. Temporal Convolutional Network (TCN)
A TCN uses **1D dilated causal convolutions** instead of recurrent connections.  
This allows it to:
- Capture long-term dependencies via dilation.
- Process sequences in parallel.
- Avoid vanishing gradient issues common in RNNs.

Unlike RNNs, TCNs do not require sequential computation, leading to faster training and inference.

---

## Dataset

The dataset is synthetically generated:
- Time values range from 0 to 200 with step size 0.1.
- The signal combines sine waves of random frequencies and amplitudes.
- Random Gaussian noise is added to introduce variability.

Each data point depends on the previous sequence of 20 time steps (sequence length = 20).

---

## Training Setup

- **Loss Function:** Huber Loss  
- **Optimizer:** Adam  
- **Learning Rate:** 3e-4  
- **Epochs:** 500  
- **Batching:** Entire dataset used in each epoch (suitable for small-scale)

---

## Evaluation and Visualization

Each model predicts future values for the test set.  
Plots display:
- Actual vs. Predicted values for both training and testing regions.
- Shaded area to indicate the test portion of the data.

Performance is measured using the test loss (Huber Loss value).

---


## Requirements

- Python ≥ 3.8  
- PyTorch  
- NumPy  
- Pandas  
- Matplotlib  
- scikit-learn  

Install dependencies:
```bash
pip install torch numpy pandas matplotlib scikit-learn
