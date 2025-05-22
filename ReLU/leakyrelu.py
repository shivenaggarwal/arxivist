import numpy as np
import matplotlib.pyplot as plt

def leakyrelu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * x)

x = np.linspace(-10, 10, 500)
y = leakyrelu(x)

plt.title("Leaky ReLU (α = 0.1)")
plt.xlabel("x")
plt.ylabel("Leaky ReLU(x)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.plot(x, y, color='green', linewidth=2)
plt.show()
