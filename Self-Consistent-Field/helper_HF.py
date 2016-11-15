# A simple Psi 4 input script to compute MP2 from a RHF reference
# Requirements scipy 0.13.0+ and numpy 1.7.2+
#
# Algorithms were taken directly from Daniel Crawford's programming website:
# http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming
# Special thanks to Rob Parrish for initial assistance with libmints
#
# Created by: Daniel G. A. Smith
# Date: 7/29/14
# License: GPL v3.0
#

import time
import numpy as np
import psi4
np.set_printoptions(precision=5, linewidth=200, suppress=True)


class helper_HF(object):

    def __init__(self, mol, memory=2, ndocc=None, scf_type='DF', guess='core'):

        # Build and all 2D values
        print('Building rank 2 integrals...')
        t = time.time()
        psi4.core.set_active_molecule(mol)
        wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
        self.wfn = wfn
        self.mints = psi4.core.MintsHelper(wfn.basisset())
        self.enuc = mol.nuclear_repulsion_energy()

        self.S = np.asarray(self.mints.ao_overlap())
        self.V = np.asarray(self.mints.ao_potential())
        self.T = np.asarray(self.mints.ao_kinetic())
        self.H = self.T + self.V

        self.Da = None
        self.Db = None
        self.Ca = None
        self.Cb = None

        self.J = None
        self.K = None

        # Orthoganlizer
        A = self.mints.ao_overlap()
        A.power(-0.5, 1.e-14)
        self.A = np.asarray(A)

        # Get nbf and ndocc for closed shell molecules
        self.epsilon = None
        self.nbf = self.S.shape[0]
        if ndocc:
            self.ndocc = ndocc
        else:
            self.ndocc = int(sum(mol.Z(A) for A in range(mol.natom())) / 2)

        # Only rhf for now
        self.nvirt = self.nbf - self.ndocc

        print('\nNumber of occupied orbitals: %d' % self.ndocc)
        print('Number of basis functions: %d' % self.nbf)

        self.C_left = psi4.core.Matrix(self.nbf, self.ndocc)
        self.npC_left = np.asarray(self.C_left)

        if guess.upper() == 'CORE':
            Xp = self.A.dot(self.H).dot(self.A)
            e, C2 = np.linalg.eigh(Xp)
            self.Ca = self.A.dot(C2)
            self.npC_left[:] = self.Ca[:, :self.ndocc]
            self.epsilon = e
            self.Da = np.dot(self.npC_left, self.npC_left.T)
        elif guess.upper() == 'SAD':
            # Cheat and run a quick SCF calculation
            psi4.set_options({'E_CONVERGENCE':1,
                            'D_CONVERGENCE':1})
            e, wfn = psi4.energy('SCF', return_wfn=True)
            self.Ca = np.array(wfn.Ca())
            self.npC_left[:] = self.Ca[:, :self.ndocc]
            self.epsilon = np.array(wfn.epsilon_a())
            self.Da = np.dot(self.npC_left, self.npC_left.T)
            psi4.set_options({'E_CONVERGENCE':6,
                             'D_CONVERGENCE':6})

        else:
            raise Exception("Guess %s not yet supported" % (guess))

        self.DIIS_error = []
        self.DIIS_F = []

        scf_type = scf_type.upper()
        if scf_type not in ['DF', 'PK', 'DIRECT', 'OUT_OF_CORE']:
            raise Exception('SCF_TYPE %s not supported' % scf_type)
        psi4.set_options({'SCF_TYPE':scf_type})
        self.jk = psi4.core.JK.build(wfn.basisset())
        self.jk.initialize()
#        self.jk.C_left().append(self.C_left)

        print('...rank 2 integrals built in %.3f seconds.' % (time.time() - t))

    def set_Cleft(self, C):
        if (C.shape[1] == self.ndocc):
            Cocc = C
        elif (C.shape[1] == self.nbf):
            self.Ca = C
            Cocc = C[:, :self.ndocc]
        else:
            raise Exception("Cocc shape is %s, need %s." % (str(self.npC_left.shape), str(Cocc.shape)))
        self.npC_left[:] = Cocc
        self.Da = np.dot(Cocc, Cocc.T)

    def diag(self, X, set_C=False):
        """
        Diaganolize with orthogonalizer A.
        """
        Xp = self.A.dot(X).dot(self.A)
        e, C2 = np.linalg.eigh(Xp)
        C = self.A.dot(C2)
        if set_C:
            self.set_Cleft(C)
        return e, C

    def build_fock(self):
        self.jk.C_left_add(self.C_left)
        self.jk.compute()
        self.jk.C_clear()
        self.J = np.asarray(self.jk.J()[0])
        self.K = np.asarray(self.jk.K()[0])
        self.F = self.H + self.J * 2 - self.K
        return self.F

    def build_jk(self, C_left, C_right=None):
        lmat = psi4.core.Matrix(C_left.shape[0], C_left.shape[1])
        np_lmat = np.asarray(lmat)
        np_lmat[:] = C_left
        self.jk.C_left_add(lmat)

        if C_right is not None:
            rmat = psi4.core.Matrix(C_right.shape[0], C_right.shape[1])
            np_rmat = np.asarray(rmat)
            np_rmat[:] = C_right
            self.jk.C_right_add(rmat)

        self.jk.compute()
        J = np.asarray(self.jk.J()[0])
        K = np.asarray(self.jk.K()[0])

        self.jk.C_clear()
        del lmat
        del np_lmat

        if C_right is not None:
            del rmat
            del np_rmat
        return J, K

    def compute_hf_energy(self):
        self.scf_e = np.einsum('ij,ij->', self.F + self.H, self.Da) + self.enuc
        return self.scf_e


class DIIS_helper(object):

    def __init__(self, max_vec=6):
        self.error = []
        self.vector = []
        self.max_vec = max_vec

    def add(self, matrix, error):
        if len(self.error) > 1:
            if self.error[-1].shape[0] != error.size:
                raise Exception("Error vector size does not match previous vector.")
            if self.vector[-1].shape != matrix.shape:
                raise Exception("Vector shape does not match previous vector.")

        self.error.append(error.ravel().copy())
        self.vector.append(matrix.copy())

    def extrapolate(self):
        # Limit size of DIIS vector
        diis_count = len(self.vector)

        if diis_count == 0:
            raise Exception("DIIS: No previous vectors.")
        if diis_count == 1:
            return self.vector[0]

        if diis_count > self.max_vec:
            # Remove oldest vector
            del self.vector[0]
            del self.error[0]
            diis_count -= 1

        # Build error matrix B
        B = np.empty((diis_count + 1, diis_count + 1))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0
        for num1, e1 in enumerate(self.error):
            B[num1, num1] = np.vdot(e1, e1)
            for num2, e2 in enumerate(self.error):
                if num2 >= num1: continue
                val = np.vdot(e1, e2)
                B[num1, num2] = B[num2, num1] = val

        # normalize
        B[abs(B) < 1.e-14] = 1.e-14
        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
        # Build residual vector
        resid = np.zeros(diis_count + 1)
        resid[-1] = -1

        # Solve pulay equations
        ci = np.dot(np.linalg.pinv(B), resid)

        # combination of previous fock matrices
        V = np.zeros_like(self.vector[-1])
        for num, c in enumerate(ci[:-1]):
            V += c * self.vector[num]

        return V

def compute_jk(psi4, jk, c_left, c_right=None):
    cl = jk.C_left()
    if not isinstance(c_left, (list, tuple)):
        c_left = [c_left]

    for c in c_left:
        mat = psi4.core.Matrix(c.shape[0], c.shape[1])
        np_mat = np.asarray(mat)
        np_mat[:] = c        
        cl.append(mat)

    if c_right is not None:
        if len(c_left) != len(c_right):
            raise ValueError("JK: length of left and right matrices is not equal")
        
        cr = jk.C_right()
        if not isinstance(c_right, (list, tuple)):
            c_right = [c_right]

        for c in c_right:
            mat = psi4.core.Matrix(c.shape[0], c.shape[1])
            np_mat = np.asarray(mat)
            np_mat[:] = c        
            cr.append(mat)

    jk.compute()
    J = []
    K = []
    for n in range(len(c_left)):
        J.append(np.array(jk.J()[n]))
        K.append(np.array(jk.K()[n]))
        del cl[0]
        if c_right is not None:
            del cr[0]

    return (J, K)


def rotate_orbitals(C, x, return_d = False):
   
    rsize = x.shape[0] + x.shape[1] 
    if (rsize) != C.shape[1]:
        raise ValueError("rotate_orbitals: shape mismatch")

    U = np.zeros((rsize, rsize))
    ndocc = x.shape[0]
    U[:ndocc, ndocc:] = x
    U[ndocc:, :ndocc] = -x.T

    U += 0.5 * np.dot(U, U)
    U[np.diag_indices_from(U)] += 1

    # Easy acess to shmidt orthogonalization
    U, r = np.linalg.qr(U.T)

    # Rotate and set orbitals
    C = C.dot(U)
    if return_d:
        Cocc = C[:, :ndocc]
        return C, np.dot(Cocc, Cocc.T)
    else:
        return C


