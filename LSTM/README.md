# LSTM (Long Short Term Memory)

A PyTorch-based word-level LSTM model trained on Friedrich Nietzsche's writings to generate philosophical-sounding text. This project explores sequence modeling, the internal structure of LSTMs, and predictive text generation.

---

## Introduction

Language is sequential. The meaning of a word often depends on what came before it. Recurrent Neural Networks (RNNs) were designed to handle such sequences by maintaining a memory of past inputs. However, they fail to capture long-term dependencies due to issues like vanishing gradients and short-term memory.

LSTMs (Long Short-Term Memory networks) were introduced to solve these problems and provide better control over the flow of information across time steps.

---

## The Core Idea Behind LSTMs

LSTMs use a memory cell and a gating mechanism to regulate information flow. These gates allow the model to retain important information and discard irrelevant or outdated data.

Each LSTM cell has three main gates:

- **Forget Gate:** Decides what information from the previous cell state should be discarded.
- **Input Gate:** Determines what new information should be added to the cell state.
- **Output Gate:** Controls what information from the current cell should be passed to the next layer or output.

---

## Mathematical Intuition

The LSTM cell is designed to preserve long-term information by controlling what gets remembered, updated, or forgotten.

Each time step operates on:

- \( x_t \): the input at time \( t \)
- \( h_{t-1} \): the previous hidden state
- \( C_{t-1} \): the previous cell state

The internal operations are:

**Forget Gate:**
Decides what portion of the previous memory \( C_{t-1} \) should be forgotten.
\[
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
\]

**Input Gate & Candidate Memory:**
Determines what new information to add to memory.
\[
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
\]
\[
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
\]

**Cell State Update:**
Blends the retained old memory and the new candidate.
\[
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
\]

**Output Gate:**
Decides what the hidden state \( h_t \) should be (used for output and next step).
\[
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
\]
\[
h_t = o_t \odot \tanh(C_t)
\]

Where:
- \( \sigma \): sigmoid activation (values between 0 and 1)
- \( \odot \): element-wise multiplication
- \( \tanh \): squashes values to [-1, 1]

**Key Insight:**
- The cell state \( C_t \) is the LSTM's internal memory.
- The gates act as regulators of information flow, enabling the model to **remember important signals and forget noise** over long sequences.
- Unlike vanilla RNNs, gradients in LSTMs propagate through the cell state more smoothly, reducing vanishing gradient issues.

---

## Limitations of LSTMs

- Computationally expensive and slower to train due to sequential processing
- Still limited in very long-range dependencies compared to attention-based models
- Less interpretable and harder to parallelize than transformers

---

## Code Summary

### 1. Preprocessing

- Load and tokenize Nietzsche’s text into lowercase words
- Build a vocabulary mapping each word to a unique index
- Convert all sentences to numerical sequences

### 2. Training Sequence Generation

- For each sentence, generate progressively increasing subsequences
- Pad all sequences to the same length so they can be batched

### 3. Dataset and Dataloader

- Inputs: all tokens except the last in a padded sequence
- Labels: the final token (word to predict)

### 4. Model Architecture

```python
class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 150, batch_first=True)
        self.fc = nn.Linear(150, vocab_size)
```

- Embedding layer maps tokens to dense vectors
- LSTM layer processes sequences and outputs hidden states
- Fully connected layer maps the final hidden state to vocabulary logits

### 5. Training
- Optimized using Adam and CrossEntropyLoss
- Trains for 50 epochs with a batch size of 32
- Monitors loss per epoch

### 6. Prediction
- Takes a user prompt like "To study physiology"
- Tokenizes and pads it
- Predicts the next word
- Feeds it back to generate further words iteratively
