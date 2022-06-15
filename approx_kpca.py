
import numpy as np
import math

from dpp import *
from kernel import *

import datetime

class ApproxKPCA:
    
    def fit(self, X, kernel, q, dpp, m=10, verbose=False):

        n = X.shape[0]
        self.X = X
        self.q = q
        self.dpp = dpp

        if verbose: print('Prepare kernel data')
        t1 = datetime.datetime.now()
        Xh, self.mu_t, self.mu_mu = compute_centered_kernel_matrix(X, kernel)
        t2 = datetime.datetime.now()
        delta = t2-t1
        if verbose: 
            print("%d secs" % delta.seconds)
            print('Prepare Approx data')
        Z = np.zeros((n, q), dtype=np.float64)
        for t in range(n):
            Z[t] = dpp.transform(Xh[t])
        t3 = datetime.datetime.now()
        delta = t3-t2
        if verbose:
            print("%d secs" % delta.seconds) 
            print('Eig Z')
        cov = np.eye(q, dtype=np.float64) * 1e-8
        for t in range(n):
            z = Z[t].reshape((q,1))
            cov = cov + np.dot(z, z.T)

        t4 = datetime.datetime.now()
        delta = t4-t3
        print("%d secs" % delta.seconds)

        # eig
        eval,evect = np.linalg.eig(cov)

        if m==-1:
            m = q

        sorted_idx = np.argsort(-eval)
        self.eval  = np.zeros(m, dtype=np.float64)
        self.evect = np.zeros((m, n), dtype=np.float64)
        for j in range(m):
            i = sorted_idx[j]
            self.eval[j] = math.sqrt(eval[i])
            u = evect[:, i].copy() # shape = n, copy to produce contiguous array
            
            ZTu = np.array([np.matmul(Z[t], u) for t in range(n)], dtype=np.float64) # shape = n
            for t in range(n):
                self.evect[j,t] += (ZTu[t]*Xh[t])[0]

        t5 = datetime.datetime.now()
        delta = t5-t4
        print("%d secs" % delta.seconds)

    def projection(self, x, axis):
        n = self.X.shape[0]
        tab = np.array([self.kernel(self.X[t], x) for t in range(n)])
        new_mu_t = np.mean(tab)
        tab = tab - new_mu_t - self.mu_t + self.mu_mu

        proj = np.zeros((len(axis),n), dtype=np.float64)
        for i in range(len(axis)):
            proj[i] = np.matmul(tab, self.evect[axis[i]]) / math.sqrt(self.eval[axis[i]])

# #############################################################################

# import pickle
# import argparse
# import matplotlib.pyplot as plt

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument('q', type=int)
    
#     args = parser.parse_args()

#     (X, _), (_, _) = tf.keras.datasets.mnist.load_data()

#     scaler = sklearn.preprocessing.StandardScaler()
#     X = X.astype(np.float32).reshape((X.shape[0], X.shape[1]*X.shape[2]))
#     #X = X[:1000]
#     X = scaler.fit_transform(X)
#     n = X.shape[0]
#     d = X.shape[1]

#     q = args.q

#     print(n, d, q)

#     kernel = rbf_kernel(0.0001)
#     kname = 'RBF'

#     W = np.random.normal(loc=0, scale=1, size=(n,q)) # shape nxq
#     W = W / math.sqrt(q) 

#     #dpp   = GaussianRandomProjection(n, q)
#     #dpp  = FeatureHashing(n, q//5, 5)
#     kpca  = ApproxKPCA()
#     mu_t, mu_mu = kpca.fit( X, n, q, kernel, W)
#     m = kpca.evect.shape[0]


#     ############################################################################
#     # plot some outputs
#     plt.plot(kpca.eval)
#     plt.savefig('akpca.%s_%d_eval.png' % (kname,q))
#     plt.clf()
#     print('eval save')

#     c = np.cumsum(kpca.eval) / np.sum(kpca.eval)
#     plt.plot(c)
#     plt.savefig('akpca.%s_%d_cumsum.png' % (kname,q))
#     print('cumsum save')

#     proj = approx_kpca_projection(X, kernel, n, q, mu_t, mu_mu, 
#     np.array([kpca.evect[0], kpca.evect[1], kpca.evect[m-1]], dtype=np.float64),
#     np.array([kpca.eval[0],kpca.eval[1],kpca.eval[m-1]], dtype=np.float64))

#     fig, ax = plt.subplots()
#     ax.scatter(proj[0], proj[1], s=1, c = 'blue', alpha=0.1)
#     plt.savefig('akpca.%s_%d_2d_01.png' % (kname, q))
#     plt.clf()
#     print('01 save')

#     fig, ax = plt.subplots()
#     ax.scatter(proj[0], proj[2], s=1, c = 'blue', alpha=0.1)

#     plt.savefig('akpca.%s_%d_2d_0l.png' % (kname, q))
#     plt.clf()
#     print('0l save')
  