
import numpy as np

from kernel import *
from kpca import *

import datetime

def nystrom_reorder(X, kernel, n, q, method):
    if method==0 or method>2: # random
        idx = np.arange(n)
        np.random.shuffle(idx)

    else:
        # compute sampling proba
        p = np.zeros(n, dtype=np.float64)
        for i in range(n):
            xi = X[i]
            if method==1: # diagonal
                k = kernel(xi, xi)
                p[i] = k**2

            else: # ==2 column-norm
                for j in range(n):
                    k = kernel(xi, X[j])
                    p[i] += k**2

        # normalized to be proba
        s = np.sum(p)
        p = p/s

        # the do the selection + ranking the remaining
        c = np.random.choice(n, size=q, replace=False, p=p)
        idx = list(c)
        for i in range(n):
            if i not in c:
                idx.append(i)
        idx = np.array(idx)

    return X[idx] # return reorder examples


class Nystrom:

    def fit(self, X, kernel, q, method=1, m=10, verbose=False):

        n = X.shape[0]
        self.X = nystrom_reorder(X, kernel, n, q, method)
        self.kernel = kernel
        self.q      = q

        # perform KPCA on q first vectors
        # >>here we use q first examples to estimate the center in kernel space
        kpca = KPCA()
        _, mu_tq, self.mu_mu = kpca.fit(self.X, kernel, d, q) # size = q 
        if verbose: print("sub-KPCA done")

        self.mu_t = np.zeros(n, dtype=np.float64)
        self.mu_t[:q] = mu_tq
        for i in range(q,n):
            tab = np.array([kernel(self.X[t], self.X[i]) for t in range(q)], dtype=np.float64)
            self.mu_t[i] = np.mean(tab)

        if m==-1:
            m = q

        if verbose: print("Evect conversion")
        t1 = datetime.datetime.now()
        self.eval  = kpca.eval.copy() * n/q
        self.evect = np.zeros((m, n), dtype=np.float64)
        for t in range(m): # !!!):
            ut = kpca.evect[t].copy()
            for i in range(n):
                k_val = np.array([kernel(X[i], X[j]) for j in range(q)]) # size = q
                mu_i  = np.mean(k_val)
                k_val = k_val - self.mu_t[:q] - mu_i + self.mu_mu
                
                self.evect[t, i] = np.sum(k_val * ut)

            self.evect[t] = self.evect[t] * math.sqrt(n/q) / kpca.eval[t]

        t2 = datetime.datetime.now()
        delta = t2-t1
        if verbose:
            print("%d secs" % delta.seconds)
            print(self.evect.shape)


    def projection(self, x, axis):

        n = self.X.shape[0]
        tab = np.array([self.kernel(self.X[t], x) for t in range(n)], dtype=np.float64)
        new_mu_t = np.mean(tab[:self.q]) # use q first to estimate center
        tab = tab - new_mu_t - self.mu_t + self.mu_mu

        proj = np.zeros((len(axis), n), dtype=np.float64)
        for i in range(len(axis)):
            proj[i] = np.dot(tab, self.evect[i]) / math.sqrt(self.eval[i])

        return proj

