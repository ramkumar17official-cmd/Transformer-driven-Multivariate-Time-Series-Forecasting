# Transformer-driven Multivariate Time Series Forecasting

This project implements a Transformer-based deep learning model for **multivariate time series forecasting** using a synthetic high-dimensional dataset.  
The project also includes a baseline LSTM model, a custom attention mechanism that exposes actual attention scores, multi-step forecasting, and complete evaluation metrics.

The goal is to demonstrate how attention-based Transformer architectures can capture long-range temporal dependencies more effectively than RNN baselines.

---

## 🚀 Project Objectives

- Create a synthetic multivariate time series dataset with:
  - Trend  
  - Multiple seasonalities  
  - Noise  
  - 20 features  
- Prepare supervised learning sequences using sliding windows.  
- Implement a Transformer model with:
  - Positional encoding  
  - Multi-head self-attention  
  - Feed-forward projection layers  
  - Multi-step forecasting (predict next 10 timesteps)  
- Implement a **custom MultiHeadAttention layer** that returns actual attention weights.  
- Train and compare:
  - Transformer model  
  - LSTM baseline model  
- Evaluate forecasting accuracy with regression metrics.  
- Plot predictions and visualize learned attention patterns.

---

## 📦 Dataset

A fully synthetic dataset is generated programmatically to ensure:
- Stationary + non-stationary components  
- Multiple periodicities  
- Realistic temporal behavior  
- Noise injection  
- 20 correlated features

Each feature has its own noise and small scaling variations on the shared underlying signal.

---

## 🧠 Modeling Approach

### 1. Positional Encoding  
Since Transformers lack recurrence, positional encoding injects temporal order into the model.

### 2. Custom Multi-Head Attention  
Unlike Keras’ built-in attention (which does NOT expose attention scores), this project includes a custom attention layer:

- Wraps Keras MultiHeadAttention  
- Forces `return_attention_scores=True`  
- Stores attention matrices for visualization  

### 3. Transformer Block  
Each block includes:
- Multi-Head Attention  
- Residual connection  
- Layer normalization  
- 2-layer Feed-Forward Network  
- Flatten and Dense layers for multi-step output

### 4. LSTM Baseline  
A single-layer LSTM is used to compare performance with a more traditional model.

---

## 📊 Metrics

The following regression metrics are used:

- **RMSE** – Root Mean Squared Error  
- **MAE** – Mean Absolute Error  
- **MAPE** – Mean Absolute Percentage Error  

---

## 🧪 Evaluation

Both models (Transformer & LSTM) are tested on unseen data.

Key results include:

- Comparison of forecasting curves  
- Quantitative performance on RMSE/MAE/MAPE  
- Visualization of attention matrices to understand:
  - Which timesteps the model focuses on  
  - Long-range context usage  
  - Patterns of internal temporal reasoning  

---

## 📁 Project Structure

```plaintext
project/
│── main.ipynb                     # Full Google Colab notebook
│── README.md
│── requirements.txt
```

---

## 🧑‍💻 How to Run in Google Colab

1. Open `main.ipynb` in Google Colab.  
2. Run all cells in sequential order:
   - Data generation  
   - Preprocessing  
   - Model building  
   - Transformer training  
   - LSTM training  
   - Evaluation  
   - Visualization  
3. Inspect:
   - Loss curves  
   - Prediction plots  
   - Attention heatmaps  

---

## 📌 Results Summary

- Transformer consistently learns global temporal patterns better than LSTM.  
- Attention maps reveal which input timesteps contribute most to forecasting.  
- Multi-step forecasts remain stable across 10-step horizons.  
- Transformer outperforms LSTM in nearly all metrics.

---

## 📑 Deliverables Included

- Fully working Google Colab notebook  
- Transformer model implementation  
- Custom attention layer  
- Data generation and processing pipeline  
- Baseline comparisons  
- Visual explanations and attention heatmaps  

---

## 👤 Author  
Ram kumar 
Cultus Internship Project — 2025
