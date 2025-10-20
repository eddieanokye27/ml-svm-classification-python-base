import numpy as np
from A2codes import minHinge, classify

# Generate a simple linearly separable dataset
np.random.seed(0)
n = 20
d = 2

# Two Gaussian blobs
X_pos = np.random.randn(n//2, d) + np.array([2, 2])
X_neg = np.random.randn(n//2, d) + np.array([-2, -2])
X_train = np.vstack((X_pos, X_neg))
y_train = np.hstack((np.ones(n//2), -np.ones(n//2)))

# Train using minHinge
lamb = 1.0
w, w0 = minHinge(X_train, y_train, lamb)

print("Learned weights:", w)
print("Learned bias:", w0)

# Predict on training data
y_pred = classify(X_train, w, w0)
train_acc = np.mean(y_pred.flatten() == y_train)

print("Training accuracy:", train_acc)
