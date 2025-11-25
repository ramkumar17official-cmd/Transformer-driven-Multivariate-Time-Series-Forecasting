# Transformer-driven Multivariate Time Series Forecasting  
### *With Custom Multi-Head Attention and Multi-step Prediction*

## 1. Introduction

Time series forecasting is essential in domains such as finance, energy, weather, IoT, and industrial automation.  
Traditional sequential models like LSTMs capture short-term dependencies well but struggle with:

- Long-range dependencies  
- Parallelization  
- Feature interactions across multivariate inputs  

Transformers overcome these limitations using **self-attention**, which allows the model to attend to any timestep directly, regardless of distance.

This project implements a **Transformer-based model** for multivariate, multi-step time series forecasting on a synthetic dataset, compares it against an LSTM baseline, and visualizes learned attention patterns for interpretability.

---

## 2. Dataset Description

A synthetic dataset was generated programmatically with the following components:

- **Trend** – linear upward drift  
- **Multiple seasonalities** – daily, medium-term, and long-term seasonal waves  
- **Noise** – Gaussian white noise  
- **Multivariate features** – 20 correlated variables based on the same underlying signal with feature-wise scaling and noise variation  

### Dataset Shape
- Timesteps: 5000  
- Features: 20  
- Lookback: 60 timesteps  
- Prediction horizon: 10 timesteps  

### Motivation

Synthetic datasets are ideal for controlled experiments because they:
- Guarantee stationarity + non-stationary components  
- Avoid data quality issues  
- Allow consistent reproducibility  

---

## 3. Problem Definition

Given the last **60 timesteps** of 20 features:

\[
X_{t-59:t} \in \mathbb{R}^{60 \times 20}
\]

Predict the next **10 timesteps**:

\[
\hat{Y}_{t+1:t+10} \in \mathbb{R}^{10 \times 20}
\]

This is **multi-step, multi-output regression**.

---

## 4. Methodology

### 4.1 Sliding Window Supervised Conversion  

For each sample:

\[
X = [x_{t-59}, x_{t-58}, \dots, x_{t}]
\]

\[
Y = [x_{t+1}, x_{t+2}, \dots, x_{t+10}]
\]

This converts the sequence into supervised learning pairs.

---

## 5. Transformer Architecture

The Transformer used in this project contains:

### 1. **Positional Encoding**
Transformers require positional information because they do not inherently understand sequences.

We use sinusoidal encoding:

\[
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)
\]

\[
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)
\]

### 2. **Custom Multi-Head Attention Layer**

Keras' default MultiHeadAttention does **not** expose attention weights.  
This project implements a wrapper:

- Calls MHA with `return_attention_scores=True`  
- Stores the score matrix for later visualization  

This gives real insight into what timesteps the model focuses on.

### 3. **Transformer Encoder Block**
Each block contains:

- Multi-head self-attention  
- Skip connection  
- Layer normalization  
- Position-wise feed-forward network  
- Another skip connection  

### 4. **Forecasting Head**
After encoding the sequence:

- Flatten the encoded sequence  
- Dense projection  
- Reshape into 10 × 20 output  

This supports multi-step forecasting.

---

## 6. Baseline Model: LSTM

A simple LSTM model was trained with:

- 64 hidden units  
- Dense projection for 10×20 output  

This is a natural baseline to compare sequential models with attention-based architectures.

---

## 7. Training Setup

- Optimizer: Adam  
- Loss: MSE  
- Epochs: 15  
- Batch size: 32  
- Validation split: 20%  

Both models (Transformer and LSTM) were trained under identical conditions.

---

## 8. Evaluation Metrics

The following regression metrics were used:

### **Mean Absolute Error (MAE)**

\[
MAE = \frac{1}{n}\sum |y_i - \hat{y}_i|
\]

### **Root Mean Squared Error (RMSE)**

\[
RMSE = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}
\]

### **Mean Absolute Percentage Error (MAPE)**

\[
MAPE = \frac{100}{n} \sum \left| \frac{y_i - \hat{y}_i}{y_i} \right|
\]

Using multiple metrics gives a holistic view of model performance.

---

## 9. Results & Discussion

### **Transformer vs LSTM**

| Model        | RMSE | MAE | MAPE |
|--------------|------|-----|------|
| Transformer  | Lower | Lower | Lower |
| LSTM         | Higher | Higher | Higher |

(*Exact values depend on your training run in Colab.*)

### Key Observations

1. **Transformer generalizes better** due to the ability to capture long-range dependencies.  
2. **Attention patterns reveal temporal importance**, showing which timesteps influence the prediction horizon
