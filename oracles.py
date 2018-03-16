import numpy as np
import scipy
from scipy.sparse import spdiags


class BinaryHinge():
    
    def __init__(self, C=1.0):

        self.C = C
     
    def func(self, X, y, w):
        
        if scipy.sparse.issparse(X):
            k = - y * X.dot(w.T) + 1
            s = np.sum(k[k > 0.0])
        else:
            k = - y * np.dot(X, w) + 1
            s = np.sum(k[k > 0.0])

        return (np.linalg.norm(w[1: ]) ** 2) / 2 + self.C * s / X.shape[0]
        
    def grad(self, X, y, w):
        
        if scipy.sparse.issparse(X):
            k = - y * X.dot(w.T) + 1
            t = spdiags(- y, 0, X.shape[0], X.shape[0]).dot(X)
            t[k <= 0] = np.zeros((1, t.shape[1]))
        else:
            k = - y * np.dot(X, w) + 1
            t = np.diag(- y).dot(X)
            t[k <= 0] = np.zeros((1, t.shape[1]))
        
        s = np.sum(t, axis=0) * self.C / X.shape[0]
        w[0] = 0
        return w + s