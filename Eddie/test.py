import numpy as np
from A2codes import minExpLinear  # or just use it in the same script

# Features (n x d)
X = np.array([
    [1, 2],
    [2, 1],
    [-1, -2],
    [-2, -1]
])

# Labels (+1 or -1)
y = np.array([1, 1, -1, -1])

# Regularization parameter
lamb = 0.1

w, w0 = minExpLinear(X, y, lamb)
print("Optimal weights:", w)
print("Optimal intercept:", w0)

def predict(X, w, w0):
    return np.sign(X.dot(w) + w0)

y_pred = predict(X, w, w0)
print("Predictions:", y_pred)
print("True labels:", y)

accuracy = np.mean(y_pred == y)
print("Accuracy:", accuracy)

