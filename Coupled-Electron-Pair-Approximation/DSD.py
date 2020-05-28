"""
Class to perform DIIS extrapolation on a sum of vectors.
- DIIS equations & algorithms from [Sherrill:1998], [Pulay:1980:393], & [Pulay:1969:197]

__authors__   =  "Jonathon P. Misiewicz"
__credits__   =  ["Jonathon P. Misiewicz"]

__copyright__ = "(c) 2014-2020, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
"""

import functools
import itertools
import numpy as np

class DirectSumDiis():
    """
    A class used to perform a DIIS extrapolation on a single vector or a list of vectors.
    If a list is given, DirectSumDiis will DIIS extrapolate the direct sum of the vectors.
    """

    def __init__(self, min_diis_vecs, max_diis_vecs):
        """
        Parameters
        ----------
        min_diis_vecs : int
            What is the minimum number of DIIS vectors to use? Any lower, and we don't extrapolate.
        max_diis_vecs : int
            What is the maximum number of DIIS vectors to use? Any more, and we throw out the oldest.
        """
        self.min = min_diis_vecs
        self.max = max_diis_vecs
        self.residuals = []
        self.trials = []

    def diis(self, r, t): 
        """
        Perform a DIIS extrapolation. r, t, and t_new should always have the same number of elements,
        if given as lists.

        Parameters
        ----------
        r : np.ndarray or list of np.ndarray
            The residual vector(s) of the latest iteration.
        t : np.ndarray or list of np.ndarray
            The amplitude vector(s) of the latest iteration.

        Returns
        -------
        t_new : np.ndarray or list of np.ndarray
            The new amplitude vector(s) after DIIS extrapolation.
        """ 

        self.residuals.append(copier(r))
        self.trials.append(copier(t))
        if len(self.residuals) > self.max:
            # Too many DIIS vectors! Get rid of the last one.
            self.residuals.pop(0)
            self.trials.pop(0)
        if len(self.residuals) >= self.min:
            # We have enough DIIS vectors to extrapolate.
            B_dim = 1 + len(self.residuals)
            B = np.empty((B_dim, B_dim))
            B[-1, :] = B[:, -1] = -1
            B[-1, -1] = 0 
            for i, ri in enumerate(self.residuals):
                for j, rj in enumerate(self.residuals):
                    if i > j: continue
                    B[i, j] = B[j, i] = direct_sum_dot(ri, rj) 
            # Normalize the B matrix to improve convergence
            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
            rhs = np.zeros((B_dim))
            rhs[-1] = -1
            coeffs = np.linalg.solve(B, rhs)[:-1]
            # Return the desired linear combination of trial vectors.
            trials_to_new = functools.partial(np.tensordot, b=coeffs, axes=(0,0))
            if isinstance(self.trials[0], np.ndarray):
                # Case: The user passed in and expects back a single array.
                return trials_to_new(self.trials)
            else:
                # Case: The user passed in and expects back a list of vectors.
                return list(map(trials_to_new, zip(*self.trials)))
        else:
            # Not enough DIIS vectors to extrapolate! Don't change the last one.
            return t

def copier(obj):
    """
    Given an np.ndarray or list of them, return copies of them.

    Parameters
    ----------
    obj : np.ndarray or list of np.ndarray

    Returns
    -------
    copy : np.ndarray or list of np.ndarray
    """

    return [i.copy() for i in obj] if not isinstance(obj, np.ndarray) else obj.copy()

def direct_sum_dot(r1, r2):
    """
    Parameters
    ----------
    r1, r2 : np.ndarray or list of np.ndarray
        The vectors or lists of vectors to compute the dot product of.

    Returns
    -------
    value : int
        The sum of the dot product for each vector in the list
    """
    if not isinstance(r1, np.ndarray):
        # Sum the dot product of each vector space in our direct sum.
        return sum(itertools.starmap(np.vdot, zip(r1, r2)))
    else:
        # There's only one vector space in our direct sum. Dot product away.
        return np.vdot(r1, r2)
