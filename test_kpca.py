
import numpy as np
import pickle
import math
import argparse
import matplotlib.pyplot as plt
import sklearn
from sklearn import *
import glob
import cv2

from kernel import *
from kpca import *
from dpp import *
from approx_kpca import *
from nystrom import *

def load_data():
    filenames = glob.glob('orl/*/*.pgm')
    np.random.shuffle(filenames) # !!!
    X = []
    for f in filenames:
      img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
      X.append(img.reshape(img.shape[0] * img.shape[1]))

    X = np.array(X).astype(np.float32)
    X = sklearn.preprocessing.StandardScaler().fit_transform(X)

    return X

######################################################################
def update_diff(eval1, eval2, diff):
    for i in range(10):
        d = abs(eval1[i] - eval2[i]) / eval1[i]
        diff[i] += d

def update_sim(evect1, evect2, sim):
    for i in range(10):
        d11 = np.sum(evect1[i]*evect1[i])
        d22 = np.sum(evect2[i]*evect2[i])
        d12 = np.sum(evect1[i]*evect2[i])

        d = abs(d12) / math.sqrt(d11 * d22)
        # print("%d %.5f" % (i, d))

        sim[i] += d

import pickle
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=int)
    parser.add_argument('q'     , type=int)
    
    args = parser.parse_args()

    X = load_data()
    n = X.shape[0]
    d = X.shape[1]

    q = args.q

    mm = 2 if args.method==0 else 3
    
    for m in range(mm):
        for k in range(4):
            print('='*30)
            if k==0:
                kernel = tanh_kernel(0.0001, 0)
                kname = 'tanh'
            elif k==1: 
                kernel = rbf_kernel(0.0001)
                kname = 'rbf'
            elif k==2:
                kernel = poly_kernel(5)
                kname = 'poly'
            elif k==3:
                kernel = sum_kernel3(tanh_kernel(0.0001, 0), 1, 
                rbf_kernel(0.0001), 1,
                poly_kernel(5), 1)
                kname = 'sum'

            kpca1 = KPCA()
            K, mu_t, mu_mu = kpca1.fit(X, kernel, d, n) 
            sim, diff = np.zeros(10), np.zeros(10)
            for _ in range(10):

                if args.method == 0: # Approx-KPCA
                    dpp = GaussianRandomProjection(n, q) if m==0 else FeatureHashing(n, q//5, 5)
                    kpca2 = ApproxKPCA()
                    mu_t, mu_mu = kpca2.fit( kernel, X, d, n, "Xh", "Z", dpp)

                else: # Nystrom
                    kpca2 = Nystrom()
                    mu_t, mu_mu = kpca2.fit(X, kernel, d, n, q, method=m, m=10)

                update_sim(kpca1.evect, kpca2.evect, sim)
                update_diff(kpca1.eval, kpca2.eval, diff)

            sim = sim/10.0
            diff = diff/10.0

            file = open('toplot/%s_%s_%d_%d' % (kname, 'akpca' if args.method==0 else 'nystrom', q, m), 'w')
            for i in range(10):
                file.write('%d,%.4f,%.4f\n' % (i, sim[i], diff[i]))
            file.close()