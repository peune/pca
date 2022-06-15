
import numpy as np
import math

from kernel import *
import datetime

#####################################################

class KPCA:

    def fit(self, X, kernel, m=-1, verbose=False): 

        n = X.shape[0]
        self.X = X
        self.kernel = kernel

        t1 = datetime.datetime.now()
        self.K, self.mu_t, self.mu_mu = compute_centered_kernel_matrix(X, kernel, n)
        t2 = datetime.datetime.now()
        delta = t2-t1
        if verbose: print("%d secs" % delta.seconds)

        eval,evect = np.linalg.eig(self.K)

        if m==-1:
            m = eval.shape[0]

        sorted_idx = np.argsort(-eval)
        self.eval  = eval[sorted_idx]
        self.evect = np.zeros((m,n), dtype=np.float64)
        for i in sorted_idx:
            self.evect[i] = evect[:, i].copy()


    def projection(self, x, axis):
        n = self.X.shape[0]
        tab = np.array([self.kernel(self.X[t], x) for t in range(n)])
        new_mu_t = np.mean(tab)
        for i in range(n):
            tab[i] = tab[i] - new_mu_t[i] - self.mu_t[i] + self.mu_mu

        proj = np.zeros((len(axis), n), dtype=np.float64)
        for i in range(len(axis)):
            proj[i] = np.matmul(tab, self.evect[axis[i]]) / math.sqrt(self.eval[axis[i]])

        return proj


