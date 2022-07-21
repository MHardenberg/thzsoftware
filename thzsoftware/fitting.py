import numpy as np


# linear fit
def linear_fit(xs, ys):
    # solving the linear model: Y = a0 + a1*X + epsilon,
    # where epsilon in (x in R, x > 0)
    X = np.vstack((np.ones(len(xs)), xs)).T

    # using X^T @ X @ beta = X^T @ y
    XT = np.matrix.transpose(X)
    XT_X = np.matmul(XT, X)
    XT_ys = np.matmul(XT, ys)
    b, a = np.matmul(np.linalg.inv(XT_X), XT_ys)
    return a, b
