import numpy as np
import matplotlib.pyplot as plt

# Function for Locally Weighted Regression
def locally_weighted_regression(x_test, X, Y, tau):
    m = X.shape[0]
    W = np.exp(-np.sum((X - x_test)**2, axis=1) / (2 * tau**2))
    W = np.diag(W)  # Convert weights into a diagonal matrix

    # Compute theta using the weighted normal equation
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term
    theta = np.linalg.inv(X_b.T @ W @ X_b) @ (X_b.T @ W @ Y)
    
    return np.array([1, x_test.item()]) @ theta  # Ensure x_test is a scalar

# Generate synthetic dataset
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)  # Linear function with noise

# Sort X for smooth plotting
X_sorted = np.sort(X, axis=0)
Y_sorted = np.array([locally_weighted_regression(x, X, Y, tau=0.1) for x in X_sorted])

# Plot results
plt.scatter(X, Y, color='blue', label='Original Data')
plt.plot(X_sorted, Y_sorted, color='red', label='LWR Fit (tau=0.1)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Locally Weighted Regression')
plt.show()
