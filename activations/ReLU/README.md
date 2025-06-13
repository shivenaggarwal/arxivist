# ReLU Activation Functions Explained

This document explores four essential activation functions commonly used in deep learning: **ReLU**, **Leaky ReLU**, **PReLU**, and **ELU**.

---

## Table of Contents
- [ReLU](#-relu-rectified-linear-unit)
- [Leaky ReLU](#-leaky-relu-leaky-rectified-linear-unit)
- [PReLU](#-prelu-parametric-relu)
- [ELU](#-elu-exponential-linear-unit)
- [Comparison Table](#-comparison-table)

---

## ReLU (Rectified Linear Unit)

```python
def relu(x):
    return np.maximum(0, x)
```

**Mathematical Definition:**
*f(x) = max(0, x)*

### Why It's Important
- Mitigates the vanishing gradient problem
- Extremely efficient (simple threshold operation)
- Leads to faster convergence than sigmoid/tanh

### Solved Problems
- Improves training speed
- Introduces sparsity in activations
- Avoids saturation issues

### Limitations
- "Dying ReLU" problem (some neurons can stop activating)
- Not zero-centered
- Can be unstable in some training scenarios

### When to Use
- Default choice for most feedforward neural networks

---

## Leaky ReLU (Leaky Rectified Linear Unit)

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
```

**Mathematical Definition:**
```
f(x) = { x     if x > 0
        { αx   if x ≤ 0
```

### Why It's Important
- Fixes the dying ReLU issue
- Maintains gradient flow for negative inputs

### Solved Problems
- Prevents neurons from going inactive
- Retains some signal for negative values

### Limitations
- α is fixed and must be manually tuned
- Output is still not zero-centered

### When to Use
- When you experience dying ReLU neurons

---

## PReLU (Parametric ReLU)

```python
def prelu(x, alpha):
    return np.where(x > 0, x, alpha * x)
```

**Mathematical Definition:**
```
f(x) = { x     if x > 0
        { αx   if x ≤ 0  (α is learned)
```

### Why It's Important
- Learns optimal slope for negative activations
- More flexible than Leaky ReLU

### Solved Problems
- Adaptive to layer/neuron behavior
- Better performance in deeper networks

### Limitations
- Adds learnable parameters (α)
- May overfit on small datasets

### When to Use
- In large networks with enough training data

---

## ELU (Exponential Linear Unit)

```python
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
```

**Mathematical Definition:**
```
f(x) = { x            if x > 0
        { α(eˣ - 1)    if x ≤ 0
```

### Why It's Important
- Combines advantages of ReLU and Leaky ReLU
- Smooths the negative values

### Solved Problems
- Outputs closer to zero-mean
- Reduces internal covariate shift
- Smoother gradient flow for negative inputs

### Limitations
- More computationally intensive
- Needs α hyperparameter tuning

### When to Use
- When you want smooth transitions and robustness to noise

---

## Comparison Table

| Feature           | ReLU         | Leaky ReLU     | PReLU         | ELU            |
|------------------|--------------|----------------|---------------|----------------|
| Negative Slope   | 0            | Fixed (≈0.01)  | Learned α     | Exponential    |
| Dying Neurons    | Yes          | No             | No            | No             |
| Compute Cost     | Low          | Low            | Medium        | High           |
| Zero-Centered    | No           | No             | No            | Yes (approx)   |
| Parameters       | 0            | 0              | 1 per layer   | 1 (α)          |
| Best For         | General use  | Stable training| Large networks| Smooth outputs |

---

> **Tip**: There’s no “one-size-fits-all.” Choose your activation based on your dataset, architecture depth, and training behavior.
