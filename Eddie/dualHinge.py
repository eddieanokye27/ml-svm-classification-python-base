import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from A2codes import dualHinge, dualClassify, cvMnist
from A2helpers import linearKernel, polyKernel, gaussKernel

# --- Generate toy binary data ---
np.random.seed(0)
X_pos = np.random.randn(10, 2) + np.array([2, 2])
X_neg = np.random.randn(10, 2) + np.array([-2, -2])
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(10), -np.ones(10)]).reshape(-1, 1)

# --- Train SVM in dual form ---
a, b = dualHinge(X, y, lamb=0.1, kernel_func=linearKernel)

# --- Predict on same data ---
yhat = dualClassify(X, a, b, X, y, lamb=0.1, kernel_func=linearKernel)

# --- Print results ---
print("Predictions:", yhat.T)
print("Training accuracy:", np.mean(yhat.flatten() == y.flatten()))


#cvMnist Test
X_fake = np.vstack([X_pos, X_neg])
y_fake = np.hstack([np.full(10, 9), np.full(10, 4)])  # 9 for +1, 4 for -1
fake_data = np.column_stack([y_fake, X_fake])

os.makedirs("test_data", exist_ok=True)
pd.DataFrame(fake_data).to_csv("test_data/A2train.csv", header=False, index=False)

# Define kernel list
kernel_list = [
    linearKernel,
    lambda x, z: polyKernel(np.atleast_2d(x), np.atleast_2d(z), degree=2)[0,0],
    lambda x, z: gaussKernel(np.atleast_2d(x), np.atleast_2d(z), width=1.0)[0,0],
]
lamb_list = [0.1, 1.0]

cv_acc, best_lamb, best_kernel = cvMnist("test_data", lamb_list, kernel_list, k=3)

print("Cross-validation accuracy matrix:\n", cv_acc)
print("Best λ:", best_lamb)
print("Best kernel function:", best_kernel)
