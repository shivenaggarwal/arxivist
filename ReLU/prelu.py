import numpy as np
import matplotlib.pyplot as plt

def prelu(x, alpha):
    return np.where(x >= 0, x, alpha * x)
x = np.linspace(-10, 10, 500)
y = prelu(x, 0.5)  # alpha = 0.5

plt.title("Parametric ReLU (PReLU) with α = 0.5")
plt.xlabel("x")
plt.ylabel("PReLU(x)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.plot(x, y, color='blue', linewidth=2)
plt.show()
