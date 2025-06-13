import numpy as np
import matplotlib.pyplot as plt

def elu(x, alpha):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

x = np.linspace(-10, 10, 500)
y = elu(x, 0.2)

plt.title("Exponential Linear Unit (ELU)")
plt.xlabel("x")
plt.ylabel("ELU(x)")
plt.grid(True)
plt.plot(x, y, color='red', linewidth=2)
plt.show()
