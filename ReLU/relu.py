import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    """Compute the Rectified Linear Unit (ReLU) function.

    Args:
        x: Input array or scalar

    Returns:
        Output of ReLU function: max(0, x)
    """
    return np.maximum(x, 0)

# Generate input values
x = np.linspace(-10, 10, 500)  # 500 points for smoother curve
y = relu(x)

# Plot configuration
plt.figure(figsize=(8, 6))  # Set figure size
plt.title("Rectified Linear Unit (ReLU) Activation Function", fontsize=14, pad=20)
plt.xlabel("Input (x)", fontsize=12)
plt.ylabel("ReLU(x)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines

# Plot the function
plt.plot(x, y, color='red', linewidth=2.5, label='ReLU')

# Highlight important features
plt.axhline(0, color='black', linewidth=0.5)  # x-axis
plt.axvline(0, color='black', linewidth=0.5)  # y-axis
plt.legend(fontsize=12)  # Add legend

plt.tight_layout()  # Adjust layout
plt.show()
