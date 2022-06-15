
import numpy as np
import math


def compute_centered_kernel_matrix(X, kernel):
    n = X.shape[0]
    K = np.zeros((n,n), dtype=np.float64)
    mu_t = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            k = kernel(X[i], X[j])
            K[i,j] = k
            K[j,i] = k

        mu_t[i] = np.mean(K[i])

    mu_mu = np.mean(K)

    for i in range(n):
        K[i] = K[i] - mu_t[i] - mu_t + mu_mu

    return K, mu_t, mu_mu

def rbf_kernel(gamma):
    def _kernel_(x,y):
        return math.exp(-gamma * np.sum((x-y)**2))

    return _kernel_

def poly_kernel(d):
    def _poly_kernel_(x,y):
        return math.pow(np.sum(x*y), d)

    return _poly_kernel_

def tanh_kernel(a, r):
    def _kernel_(x,y):
        return math.tanh(a*np.sum(x*y) + r)

    return _kernel_

def sum_kernel(k1, c1, k2, c2):
    def _kernel_(x,y):
        return k1(x,y)*c1 + k2(x,y)*c2

    return _kernel_

def sum_kernel3(k1, c1, k2, c2, k3, c3):
    def _kernel_(x,y):
        return k1(x,y)*c1 + k2(x,y)*c2 + k3(x,y)*c3

    return _kernel_
