import numpy as np, math
import pandas as pd
import matplotlib.pyplot as plt
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

#b)
def minHinge(X, y, lamb, stabilizer=1e-5):
    n, d = X.shape
    y = y.flatten()
    
    # quadratic term (only penalize w)
    P = np.diag(np.concatenate([lamb*np.ones(d), [0], np.zeros(n)]))
    P += stabilizer * np.eye(d + 1 + n)

    # linear term
    q = np.concatenate([np.zeros(d+1), np.ones(n)])
    
    # linear coonstraints: y_i (x_i w + w0) >= 1 - xi
    G_std = np.zeros((n, d+1+n))
    for i in range(n):
        G_std[i, :d] = -y[i] * X[i]
        G_std[i, d] = -y[i]
        G_std[i, d+1 + i] = -1
    h_std = -np.ones(n)
    
    # xi >= 0  →  -xi ≤ 0
    G_xi = np.zeros((n, d+1+n))
    for i in range(n):
        G_xi[i, d+1 + i] = -1
    h_xi = np.zeros(n)
    
    # combine constraints
    G = np.vstack([G_std, G_xi])
    h = np.hstack([h_std, h_xi])
    
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    
    P_cvx = matrix(P)
    q_cvx = matrix(q)
    G_cvx = matrix(G)
    h_cvx = matrix(h)
    
    sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    z = np.array(sol['x']).flatten()
    
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
    train_acc_explinear_avg = np.mean(train_acc_explinear, axis=2)
    test_acc_explinear_avg = np.mean(test_acc_explinear, axis=2)
    train_acc_hinge_avg = np.mean(train_acc_hinge, axis=2)
    test_acc_hinge_avg = np.mean(test_acc_hinge, axis=2)

    # put ExpLinear and Hinge data into 4x6 matrices
    train_acc = np.hstack([train_acc_explinear_avg, train_acc_hinge_avg])
    test_acc = np.hstack([test_acc_explinear_avg, test_acc_hinge_avg])

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

    # Build kernel matrix correctly
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            Kij = np.atleast_2d(kernel_func(X[i:i+1], X[j:j+1]))[0, 0]
            K[i, j] = Kij
            K[j, i] = Kij

    y_col = y.reshape(-1, 1)
    DeltaKDelta = (y_col * K) * y_col.T
    P = (1.0 / lamb) * DeltaKDelta + stabilizer * np.eye(n)
    q = -np.ones(n)

    P_cvx = matrix(P)
    q_cvx = matrix(q)
    G_cvx = matrix(np.vstack([-np.eye(n), np.eye(n)]))
    h_cvx = matrix(np.hstack([np.zeros(n), np.ones(n)]))
    A_cvx = matrix(y.reshape(1, -1))
    b_cvx = matrix([0.0])

    solvers.options['show_progress'] = False
    sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
    alpha = np.clip(np.array(sol['x']).flatten(), 0, 1)

    # compute b robustly
    sv_mask = (alpha > 1e-5) & (alpha < 1 - 1e-5)
    if np.any(sv_mask):
        b_vals = []
        for i in np.where(sv_mask)[0]:
            b_vals.append(y[i] - (1.0 / lamb) * np.dot(K[i, :], y * alpha))
        b = np.mean(b_vals)
    else:
        b = 0.0

    return alpha[:, None], b


#b)

def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    Xtest = np.asarray(Xtest)
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    a = np.asarray(a).reshape(-1)

    m, n = Xtest.shape[0], X.shape[0]
    Ktest = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            Ktest[i, j] = np.atleast_2d(kernel_func(Xtest[i:i+1], X[j:j+1]))[0, 0]

    f = (1.0 / lamb) * (Ktest @ (y * a)) + b
    yhat = np.sign(f).reshape(-1, 1)
    yhat[yhat == 0] = 1.0
    return yhat






#c)

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

