
import numpy as np

from dpp import *
from pca import *

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

class ApproxPCA:

    def fit(self, X, dpp, m=10, verbose=False): # assume X is properly centered

        n = X.shape[0]
        d = X.shape[1]
        q = dpp.dim()

        if verbose: print('Approx data', flush=True)
        Z = np.zeros((n,q), dtype=np.float64)
        for i in range(n):
            Z[i] = dpp.transform(X[i])

        # construct covariance matrix of approx data Z
        if verbose: print('Covmat', flush=True)
        C = np.eye(q) * 1e-8
        for t in range(n):
            x = Z[t].reshape((q,1))
            C = C + np.matmul(x, x.T)
        C = C / n

        if verbose: print('Eig', flush=True)
        eval,evect = np.linalg.eig(C)

        # convert the eigenvectors into eigenvectors of the input covariance matrix
        if verbose: print('Conversion', flush=True)
        sorted_idx = np.argsort(-eval)
        self.eval  = eval[sorted_idx]
        self.evect = np.zeros((m,d), dtype=np.float64)
        for i in range(m):
            u = evect[:, i].copy() # q 

            v = np.zeros(n)
            for t in range(n):
                v[t] = np.matmul(Z[t], u) 
      
            for t in range(n):
                self.evect[i] += v[t]*X[t]

            self.evect[i] = self.evect[i] / (math.sqrt(np.sum(self.evect[i]**2)) + 1e-8)

    def projection(self, x, axis):
        proj = np.zeros((len(axis)))
        for i in range(len(axis)):
            proj[i] = np.matmul(x, self.evect[axis[i]])
        return proj

    def projection_data(self, X, axis):
        proj = np.zeros((len(axis), X.shape[0]))
        for i in range(len(axis)):
            proj[i] = np.matmul(X, self.evect[axis[i]])
        return proj
