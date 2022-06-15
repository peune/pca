
import numpy as np
from pca import *

import tensorflow as tf
from tensorflow import *

import sklearn
from sklearn import *

def load_data_ex1(name):
  if name == 'mnist':
    (X, _), (_, _) = tf.keras.datasets.mnist.load_data()
    # plt.imshow(X[29049], cmap='gray')
    # plt.savefig('toto.png')
    # quit()

    X = X.astype(np.float32).reshape((X.shape[0], X.shape[1]*X.shape[2]))

  elif name == 'cifar':
    (X, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    X = X.astype(np.float32).reshape((X.shape[0], X.shape[1]*X.shape[2]*3))
  
  elif name == 'diabetes':
    (X,_) = sklearn.datasets.load_diabetes(return_X_y=True)

  elif name == 'breast_cancer':
    (X,_) = sklearn.datasets.load_breast_cancer(return_X_y=True)

  elif name == 'iris':
    (X,_) = sklearn.datasets.load_iris(return_X_y=True)

  scaler = sklearn.preprocessing.StandardScaler()
  X = scaler.fit_transform(X)
  #X = X/255

  return X, scaler

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

from pca import *
from dpp import *
from approx_pca import *

import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('m', type=int)
    parser.add_argument('q', type=int)
    
    args = parser.parse_args()

    X,scaler = load_data_ex1(args.data)
    n = X.shape[0]
    d = X.shape[1]

    pca = PCA()
    pca.fit_from_cov(X)

    m = args.m 
    q = args.q

    sim, diff = np.zeros(10), np.zeros(10)
    for _ in range(10):

      dpp = GaussianRandomProjection(d, q) if m==0 else FeatureHashing(d, q//5, 5)
      apca = ApproxPCA()
      apca.fit(X, dpp, verbose=True)

      update_sim(pca.evect, apca.evect, sim)
      update_diff(pca.eval, apca.eval, diff)

    sim = sim/10.0
    diff = diff/10.0

    for i in range(10):
      print('%d,%.4f,%.4f' % (i, sim[i], diff[i]))





