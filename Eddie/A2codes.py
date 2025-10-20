import numpy as np, matplotlib, pandas, math
from scipy.optimize import minimize
from cvxopt import matrix, solvers
from A2helpers import polyKernel, linearKernel, gaussKernel, generateData

#Question 1

#a)
def minExpLinear(X, y, lamb):

    n, d = X.shape
    
    # Flatten y in case it's column vector
    y = y.reshape(-1)
    
    # Initial guess for [w, w0]
    w0_init = np.zeros(d + 1)
    
    # Define the loss function
    def loss_fn(params):
        w = params[:d]
        w0 = params[d]
        margins = y * (X.dot(w) + w0)
        
        # ExpLinear loss
        loss = np.where(margins <= 0, 1 - margins, np.exp(-margins))
        return np.sum(loss) + (lamb / 2) * np.sum(w**2)
    
    # Minimize the loss
    result = minimize(loss_fn, w0_init, method='L-BFGS-B')
    
    w_opt = result.x[:d]
    w0_opt = result.x[d]
    
    return w_opt, w0_opt

import numpy as np
from cvxopt import matrix, solvers

#b)
def minHinge(X, y, lamb, stabilizer=1e-5):
    n, d = X.shape
    y = y.flatten()
    
    # P matrix (quadratic term)
    P = np.diag(np.concatenate([lamb*np.ones(d), [0], np.zeros(n)]))
    P += stabilizer * np.eye(d + 1 + n)

    # q vector
    q = np.concatenate([np.zeros(d+1), np.ones(n)])
    
    # G and h matrices
    # 1 - y_i (x_i w + w0) <= xi
    G_std = np.zeros((n, d+1+n))
    for i in range(n):
        G_std[i, :d] = -y[i] * X[i]
        G_std[i, d] = -y[i]
        G_std[i, d+1 + i] = 1
    h_std = np.ones(n)

    # xi >= 0
    G_xi = np.zeros((n, d+1+n))
    for i in range(n):
        G_xi[i, d+1 + i] = -1
    h_xi = np.zeros(n)

    G = np.vstack([G_std, G_xi])
    h = np.hstack([h_std, h_xi])

    # convert to cvxopt matrices
    P_cvx = matrix(P)
    q_cvx = matrix(q)
    G_cvx = matrix(G)
    h_cvx = matrix(h)

    solvers.options['show_progress'] = False
    solution = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    z = np.array(solution['x']).flatten()
    
    w = z[:d]
    w0 = z[d]
    
    return w, w0


#c)
def classify(Xtest, w, w0):
    # Compute linear combination and apply sign
    yhat = np.sign(Xtest @ w + w0)
    
    # mak sure that output is a column vector (m x 1)
    return yhat.reshape(-1, 1)


#d)

def synExperimentsRegularize():
    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.]
    gen_model_list = [1, 2, 3]

    # storage
    train_acc_explinear = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_explinear = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])

    # Set random seed (change too group id)
    np.random.seed(38)

    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                # Generate synthetic training and test data
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                # ExpLinear classifier
                w, w0 = minExpLinear(Xtrain, ytrain, lamb)
                ytrain_pred = classify(Xtrain, w, w0)
                ytest_pred = classify(Xtest, w, w0)
                train_acc_explinear[i, j, r] = np.mean(ytrain_pred.flatten() == ytrain.flatten())
                test_acc_explinear[i, j, r] = np.mean(ytest_pred.flatten() == ytest.flatten())

                # Hinge classifier
                w, w0 = minHinge(Xtrain, ytrain, lamb)
                ytrain_pred = classify(Xtrain, w, w0)
                ytest_pred = classify(Xtest, w, w0)
                train_acc_hinge[i, j, r] = np.mean(ytrain_pred.flatten() == ytrain.flatten())
                test_acc_hinge[i, j, r] = np.mean(ytest_pred.flatten() == ytest.flatten())

    # Average accuracies over runs
    train_acc_explinear_avg = np.mean(train_acc_explinear, axis=2)  # shape: (4,3)
    test_acc_explinear_avg = np.mean(test_acc_explinear, axis=2)
    train_acc_hinge_avg = np.mean(train_acc_hinge, axis=2)
    test_acc_hinge_avg = np.mean(test_acc_hinge, axis=2)

    # put ExpLinear and Hinge data into 4x6 matrices
    train_acc = np.hstack([train_acc_explinear_avg, train_acc_hinge_avg])  # shape: 4x6
    test_acc = np.hstack([test_acc_explinear_avg, test_acc_hinge_avg])     # shape: 4x6

    return train_acc, test_acc




#Question 3

#a)

def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n = X.shape[0]
    if n == 0:
        raise ValueError("X must contain at least one sample")
    if lamb <= 0:
        raise ValueError("lamb must be > 0")

    # Build kernel matrix K (n x n)
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            Kij = kernel_func(X[i], X[j])
            K[i, j] = Kij
            K[j, i] = Kij

    y_col = y.reshape(-1, 1)
    DeltaKDelta = (y_col * K) * y_col.T

    P = (1.0 / lamb) * DeltaKDelta

    # stabilizer
    P = P + stabilizer * np.eye(n)
    q = -np.ones(n, dtype=float)

    #qp
    solvers.options['show_progress'] = False

    P_cvx = matrix(P)
    q_cvx = matrix(q)

    # G and h
    G_top = -np.eye(n)
    G_bottom = np.eye(n)
    G_np = np.vstack([G_top, G_bottom])
    h_np = np.hstack([np.zeros(n), np.ones(n)])

    G_cvx = matrix(G_np)
    h_cvx = matrix(h_np)

    A_cvx = matrix(y.reshape(1, -1).astype(float))
    b_cvx = matrix(np.array([0.0]))

    sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
    alpha = np.array(sol['x']).reshape(-1)

    alpha = np.clip(alpha, 0.0, 1.0)

    a = alpha.reshape(-1, 1)

    # compute intercept b:
    # choose an index i whose alpha_i is closest to 0.5 (robust to numeric)
    idx = int(np.argmin(np.abs(alpha - 0.5)))

    # compute b using chosen i: b = y_i - (1/lamb) * k_i^T (Delta(y) alpha)
    delta_y_alpha = y * alpha
    k_i = K[idx, :]
    b = float(y[idx] - (1.0 / lamb) * np.dot(k_i, delta_y_alpha))

    return a, b

#b)

def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    Xtest = np.asarray(Xtest)
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    a = np.asarray(a).reshape(-1)

    m = Xtest.shape[0]
    n = X.shape[0]

    # compute kernel matrix between test points and training points
    Ktest = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            Ktest[i, j] = kernel_func(Xtest[i], X[j])

    delta_y_a = y * a  # (n,)

    f = (1.0 / lamb) * (Ktest @ delta_y_a) + b  # (m,)

    # Predicted labels = sign(f)
    yhat = np.sign(f).reshape(-1, 1)

    # Handle case where f == 0 → classify as +1
    yhat[yhat == 0] = 1.0

    return yhat





#c)
import numpy as np
import pandas as pd

def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):

    # load dataset
    train_data = pd.read_csv(f"{dataset_folder}/A2train.csv", header=None).to_numpy()
    X = train_data[:, 1:] / 255.0        # normalize pixel values
    y = train_data[:, 0][:, None]        # (n, 1)
    y[y == 4] = -1
    y[y == 9] = 1
    y = y.astype(float)

    n = X.shape[0]

    # initalize arrays
    cv_acc = np.zeros([k, len(lamb_list), len(kernel_list)])

    # Get seed
    np.random.seed(38)

    # shuffle around data and split indices for k folds process
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)

    # Execute cross-validation process
    for i, lamb in enumerate(lamb_list):
        for j, kernel_func in enumerate(kernel_list):
            for l in range(k):
                # Get validation indices for this fold
                val_idx = folds[l]
                train_idx = np.hstack([folds[m] for m in range(k) if m != l])

                # Split train and validation sets
                Xtrain, ytrain = X[train_idx], y[train_idx]
                Xval, yval = X[val_idx], y[val_idx]

                # Train dual SVM
                a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func)

                # Predict on validation fold
                yhat = dualClassify(Xval, a, b, Xtrain, ytrain, lamb, kernel_func)

                # Compute accuracy
                acc = np.mean(yhat.flatten() == yval.flatten())
                cv_acc[l, i, j] = acc

    # Get average accuracy across folds
    mean_acc = np.mean(cv_acc, axis=0)

    # Get best hyperparams
    best_idx = np.unravel_index(np.argmax(mean_acc), mean_acc.shape)
    best_lamb = lamb_list[best_idx[0]]
    best_kernel = kernel_list[best_idx[1]]

    return mean_acc, best_lamb, best_kernel

