import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

class PCA:

    def init(self, sorted_eval, sorted_evect):
        self.eval  = sorted_eval.copy()
        self.evect = sorted_evect.copy()

    def fit_from_cov(self, X): # assume X is properly centered

        C = np.matmul(X.T, X) / X.shape[0] # construct covariance matrix
        eval,evect = np.linalg.eig(C)

        idx = np.argsort(-eval)
        self.eval  = eval[idx]
        self.evect = []
        for i in idx:
            self.evect.append( evect[:, i] )

    def fit_from_dot(self, X):

        D = np.matmul(X, X.T) / X.shape[1]
        eval,evect = np.linalg.eig(D)

        idx = np.argsort(-eval)
        self.eval  = eval[idx]
        self.evect = []
        for i in idx:
            v = evect[:, i] # shape = n
            u = np.matmul(X.T, v)
            u = u / np.sqrt(np.sum(u**2))
            self.evect.append( u )

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

