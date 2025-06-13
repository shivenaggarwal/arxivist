# Character-Level RNN Classifier

A PyTorch-based implementation of a Vanilla Recurrent Neural Network (RNN) for character-level name classification by language.

---

## Why Do We Need RNNs?

Traditional neural networks (e.g., feedforward or convolutional) process fixed-size input and assume independence between data points. However, many real-world problems involve **sequential data** where the **order of information is crucial**:

- Natural Language Processing
- Time Series Forecasting
- Audio and Speech Processing

In such scenarios, the current input depends on previous ones. RNNs address this by maintaining a **memory of past inputs** via a **hidden state**.

---

## What Problems Do RNNs Solve?

RNNs improve upon traditional architectures by:

- Accepting **variable-length sequences**
- **Sharing parameters** across time steps
- **Remembering context** using a hidden state

| Problem                          | Solution Provided by RNN              |
|----------------------------------|---------------------------------------|
| Variable input lengths           | Processes input sequentially          |
| Need for memory of past inputs   | Maintains a hidden state              |
| Static structure for dynamic data| Recurrent connections and shared weights |

---

## How RNNs Work: Architecture Overview

### Core Idea

An RNN processes an input sequence one element at a time, updating a hidden state along the way.

Input sequence:

```
x₁ → x₂ → x₃ → … → xₜ
↑     ↑     ↑        ↑
h₁   h₂   h₃       hₜ
```

At each time step `t`:

- **Input**: `xₜ`
- **Previous hidden state**: `hₜ₋₁`
- **Current hidden state**: `hₜ`
- **Output**: `yₜ`

Mathematically:

```math
hₜ = tanh(W_{ih} · xₜ + W_{hh} · hₜ₋₁ + b)
yₜ = softmax(W_{ho} · hₜ + b_{out})
```

---

## Data Preprocessing

- Each name is transformed into a **sequence of one-hot encoded characters**
- Each language label is mapped to a **category index**
- The RNN reads the name **character by character**

**Example**:

```
"Abbott" → ['A', 'b', 'b', 'o', 't', 't'] → [One-hot vectors]
```

---

## Loss and Optimization

- **Loss Function**: `Negative Log Likelihood (NLLLoss)`
- **Output**: Log-probabilities across language categories
- **Optimizer**: `Stochastic Gradient Descent (SGD)`

---

## Problems With Vanilla RNNs

| Issue                  | Description                                             |
|------------------------|---------------------------------------------------------|
| Short-Term Memory      | Struggles with long-range dependencies                 |
| Vanishing Gradients    | Gradients shrink exponentially, limiting learning       |
| Exploding Gradients    | Gradients grow too large, causing instability           |

---

## Solutions and Improvements to RNNs

| Improvement   | Paper / Year                    | Description                             |
|---------------|----------------------------------|-----------------------------------------|
| **LSTM**      | Hochreiter & Schmidhuber (1997) | Introduces gates to control memory flow |
| **GRU**       | Cho et al. (2014)               | Simplified gated variant of LSTM        |
| **Attention** | Bahdanau et al. (2014)          | Learns to focus on relevant input parts |
| **Transformer** | Vaswani et al. (2017)         | Replaces recurrence with self-attention |

---

## Forward Pass Intuition

At each time step:

1. Concatenate input `xₜ` with previous hidden state `hₜ₋₁`
2. Apply a linear transformation followed by `tanh` to update `hₜ`
3. Generate output probabilities from `hₜ`

```python
combined = torch.cat((x_t, h_prev), dim=1)
h_t = torch.tanh(W_combined @ combined + b)
y_t = F.log_softmax(W_out @ h_t + b_out, dim=1)
```

---

## Backpropagation Through Time (BPTT)

Training involves **unrolling** the network across time and performing backpropagation. The gradient is accumulated from each time step:

```math
∂L/∂W = ∑ₜ (∂Lₜ/∂yₜ) * (∂yₜ/∂hₜ) * (∂hₜ/∂W)
```

### Common Issues:
- **Vanishing gradients** → no learning for earlier steps
- **Exploding gradients** → instability

### Solutions:
- **Gradient clipping**
- **Use of LSTM/GRU**
- **Truncated BPTT**

---

## Results

- Trained on a **character-level dataset** of names and languages
- Given a name, the model predicts its **language of origin**
- Achieves **reasonable validation accuracy** using a simple RNN architecture

---

> _“A name carries culture. With RNNs, we decode it—character by character.”_
