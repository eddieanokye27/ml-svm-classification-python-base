import numpy as np
from A2codes import cvMnist
from A2helpers import linearKernel, polyKernel, gaussKernel
import pandas as pd, os

# Easy separable toy data
np.random.seed(42)
X_pos = np.random.randn(20, 2) + np.array([3, 3])
X_neg = np.random.randn(20, 2) + np.array([-3, -3])
X_fake = np.vstack([X_pos, X_neg])
y_fake = np.hstack([np.full(20, 9), np.full(20, 4)])  # 9 → +1, 4 → -1
fake_data = np.column_stack([y_fake, X_fake])

os.makedirs("test_data", exist_ok=True)
pd.DataFrame(fake_data).to_csv("test_data/A2train.csv", header=False, index=False)

kernel_list = [
    linearKernel,
    lambda x, z: polyKernel(np.atleast_2d(x), np.atleast_2d(z), degree=2)[0,0],
    lambda x, z: gaussKernel(np.atleast_2d(x), np.atleast_2d(z), width=1.0)[0,0],
]
lamb_list = [0.1, 1.0]

cv_acc, best_lamb, best_kernel = cvMnist("test_data", lamb_list, kernel_list, k=3)

print("Cross-validation accuracy matrix:\n", cv_acc)
print("Best λ:", best_lamb)
print("Best kernel:", best_kernel)
