#
# Dot Product Preserving
#

import math
import numpy as np


class GaussianRandomProjection:

  def __init__(self, d, q):
    self.d = d
    self.q = q
    self.w = np.random.normal(loc=0, scale=1, size=(q,d)) # shape qxd
    self.w = self.w / math.sqrt(q) 

    # I = np.matmul(self.w.T, self.w)
    # s = (np.sum(I) - np.trace(I)) / (d*d - d)
    # t = np.trace(I) / d
    # I = I - np.eye(d)
    # r = np.sum(I**2) / d
    # print("[%d %.5f %.5f %.5f]" % (q, s, t, math.sqrt(r)))

  def dim(self):
    return self.q

  def transform(self, x):
    return np.matmul(x, self.w.T)

    

class FeatureHashing:

  def __init__(self, d, q, m): # d=org_dim, q=low_dim of one hash, m=num_hash
    self.d = d
    self.q = q
    self.m = m
    self.h = np.random.randint(q, size=(m, d))
    self.s = np.random.binomial(n=1, p=0.5, size=(m, d))
    self.s = 2*self.s - 1

  def dim(self):
    return self.m*self.q

  def transform(self, x):
    y = np.zeros((self.m*self.q))
    for k in range(self.m):
      o = k*self.q
      for j in range(self.q):
        y[o+j] = np.sum(x[self.h[k]==j] * self.s[k, self.h[k]==j], axis=0) / self.m      

    return y
    

##########################################################################################
# def eval_dot_product(dpp_transform, Xorg, m=100):
#   X = []
#   n = 5000
#   for t in range(n):
#     X.append(Xorg[t])
#   X = np.array(X)

#   Z = dpp_transform.transform_mem(X)

#   d = X.shape[1]
#   diff = 0.
#   for k in range(m):
#     i,j = np.random.randint(Z.shape[0], size=2)
#     xx = np.sum(X[i]*X[j])
#     zz = np.sum(Z[i]*Z[j])
#     dd = abs( (xx-zz)/xx )
#     diff += dd

#   print("%.5f" % (diff/m))
#   return diff/m



# def eval_dot_product_old(dpp_transform, X, input_dataset_filename, tmp_dataset_filename, q, m=100):
#   #Z = dpp_transform.fit_transform_mem(input_dataset_filename, 5000)
#   #Z = dpp_transform.fit_transform(X)
#   print(tmp_dataset_filename)
#   dpp_transform.fit_transform(input_dataset_filename, tmp_dataset_filename, 5000)
#   Z = DatasetReader(tmp_dataset_filename)
#   nz = len(Z)

#   d = X.dim
#   diff = 0.
#   for k in range(m):
#     i,j = np.random.randint(nz, size=2)
#     xx = np.sum(X[i]*X[j])
#     zz = np.sum(Z[i]*Z[j])
#     dd = abs( (xx-zz)/xx )
#     if dd>10:
#       print("%.4f try again" % dd)
#       k = k-1
#       continue
#     diff += dd

#   print(diff)
#   return diff/m

# import matplotlib.pyplot as plt

# import argparse
# from enum import Enum

# class DPPMethod(Enum):
#     gaussian  = 1
#     feathash  = 2

#     def __str__(self):
#         return self.name

#     @staticmethod
#     def from_string(s):
#         try:
#             return DPPMethod[s]
#         except KeyError:
#             raise ValueError()


# def test_dot(X, input_dataset_filename, tmp_dataset_filename, dpp_method, numhash, l, tiks, output):

#     plt.xticks(tiks, [str(q) for q in l])

#     d = X.dim
#     res = []
#     tab_mean = []
#     tab_sd = []
#     for t,q in enumerate(l):

#         tab = []
#         for k in range(5):

#             print('.', end='', flush=True)
#             if dpp_method == DPPMethod.gaussian:
#                 dpp = GaussianRandomProjection(d, q)
#             elif dpp_method == DPPMethod.feathash:
#                 dpp = FeatureHashing(d, q, numhash)

#             #diff = eval_dot_product(dpp, X, input_dataset_filename, tmp_dataset_filename, q, m=10000)
#             diff = eval_dot_product(dpp, X, m=10000)
#             tab.append(diff)

#         mean = np.mean(tab)
#         tab_sd.append(np.std(tab))

#         tab_mean.append(mean)

#         res.append(tab)

#     res = np.array(res)

#     print(tab_mean)
#     plt.plot(l, tab_mean, color='blue')
#     plt.savefig('%s.png' % output)

#     print('')
#     print(tab_sd)

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', '--data'     , type=str, required=True)
#     parser.add_argument('-p', '--dpp'      , type=DPPMethod.from_string, choices=list(DPPMethod), required=False)
#     parser.add_argument('-n', '--numhash', type=int, default=10, required=False, help="Number of hash functions required by FeatureHashingM method")
    
#     args = parser.parse_args()

#     X= DatasetReader(args.data)

#     l = [5000, 3000, 2000, 1000, 500]
#     test_dot(X, args.data, 'tmp.dpp', args.dpp, args.numhash,
#     l, l, '%s_%s_%d_dot' % (args.data, args.dpp, args.numhash))

