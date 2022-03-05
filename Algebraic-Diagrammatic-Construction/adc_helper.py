'''
Helper functions for the ADC code. Davidson algorithm adapted from
Coupled-Cluster/RHF/EOM_CCSD.py.
'''

__authors__ = 'Oliver J. Backhouse'
__credits__ = ['T. Daniel Crawford', 'Andrew M. James', 'Oliver J. Backhouse']

__copyright__ = '(c) 2014-2020, The Psi4NumPy Developers'
__license__ = 'BSD-3-Clause'
__date__ = '2018-03-01'

import time
import numpy as np


def davidson(matrix, guesses, diag, maxiter=None, sort_via_abs=True, tol=1e-10, nvec_per_root=20):
    if callable(matrix):
        matvec = matrix
    else:
        matvec = lambda v: np.dot(matrix, v)

    if sort_via_abs:
        picker = lambda x: np.argsort(np.absolute(x))[:k]
    else:
        picker = lambda x: np.argsort(x)[:k]

    k = guesses.shape[1]
    b = guesses.copy()
    theta = np.zeros((k,))

    if maxiter is None:
        maxiter = k * 20
    
    for niter in range(maxiter):
        b, r = np.linalg.qr(b)
        theta_old = theta[:k]

        s = np.zeros_like(b)
        for i in range(b.shape[1]):
            s[:,i] = matvec(b[:,i])

        g = np.dot(b.T, s)

        theta, alpha = np.linalg.eigh(g)
        idx = picker(theta)
        theta = theta[idx]
        alpha = alpha[:,idx]

        b_new = []
        for i in range(k):
            w  = np.dot(s, alpha[:,i])
            w -= np.dot(b, alpha[:,i]) * theta[i]
            q = w / (theta[i] - diag[i] + 1e-20)
            b_new.append(q)

        de = np.linalg.norm(theta[:k] - theta_old)
        if de < tol:
            conv = True
            b = np.dot(b, alpha)
            break
        else:
            if b.shape[1] >= (k * nvec_per_root):
                b = np.dot(b, alpha)
                theta = theta_old
            else:
                b = np.concatenate([b, np.column_stack(b_new)], axis=1)

    b = b[:, :guesses.shape[-1]]

    return theta, b
