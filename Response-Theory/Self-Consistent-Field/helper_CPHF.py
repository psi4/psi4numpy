"""
Helper classes and functions for molecular properties requiring
solution of CPHF equations.

References:
- Equations and algorithms from [Szabo:1996] and Project 3 from
Daniel Crawford's programming website:
http://github.com/CrawfordGroup/ProgrammingProjects
"""

__authors__   =  "Daniel G. A. Smith"
__credits__   = ["Daniel G. A. Smith", "Eric J. Berquist"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-8-30"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

import os.path
import sys
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '../../Self-Consistent-Field'))
from helper_HF import DIIS_helper


class helper_CPHF(object):

    def __init__(self, mol, numpy_memory=2):

        self.mol = mol
        self.numpy_memory = numpy_memory

        # Compute the reference wavefunction and CPHF using Psi
        scf_e, self.scf_wfn = psi4.energy('SCF', return_wfn=True)

        self.C = self.scf_wfn.Ca()
        self.Co = self.scf_wfn.Ca_subset("AO", "OCC")
        self.Cv = self.scf_wfn.Ca_subset("AO", "VIR")
        self.epsilon = np.asarray(self.scf_wfn.epsilon_a())

        self.nbf = self.scf_wfn.nmo()
        self.nocc = self.scf_wfn.nalpha()
        self.nvir = self.nbf - self.nocc

        # Integral generation from Psi4's MintsHelper
        self.mints = psi4.core.MintsHelper(self.scf_wfn.basisset())

        # Get nbf and ndocc for closed shell molecules
        print('\nNumber of occupied orbitals: %d' % self.nocc)
        print('Number of basis functions: %d' % self.nbf)

        # Grab perturbation tensors in MO basis
        nCo = np.asarray(self.Co)
        nCv = np.asarray(self.Cv)
        self.tmp_dipoles = self.mints.so_dipole()
        self.dipoles_xyz = []
        for num in range(3):
            Fso = np.asarray(self.tmp_dipoles[num])
            Fia = (nCo.T).dot(Fso).dot(nCv)
            Fia *= -2
            self.dipoles_xyz.append(Fia)

    def run(self, method='direct', omega=None):
        self.method = method
        if self.method == 'direct':
            if not omega:
                self.solve_static_direct()
            else:
                self.solve_dynamic_direct(omega=omega)
        elif self.method == 'iterative':
            if not omega:
                self.solve_static_iterative()
            else:
                self.solve_dynamic_iterative(omega=omega)
        else:
            raise Exception("Method %s is not recognized" % self.method)
        self.form_polarizability()

    def solve_static_direct(self):
        # Run a quick check to make sure everything will fit into memory
        I_Size = (self.nbf ** 4) * 8.e-9
        oNNN_Size = (self.nocc * self.nbf ** 3) * 8.e-9
        ovov_Size = (self.nocc * self.nocc * self.nvir * self.nvir) * 8.e-9
        print("\nTensor sizes:")
        print("ERI tensor           %4.2f GB." % I_Size)
        print("oNNN MO tensor       %4.2f GB." % oNNN_Size)
        print("ovov Hessian tensor  %4.2f GB." % ovov_Size)

        # Estimate memory usage
        memory_footprint = I_Size * 1.5
        if I_Size > self.numpy_memory:
            psi4.core.clean()
            raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                            limit of %4.2f GB." % (memory_footprint, self.numpy_memory))

        # Compute electronic Hessian
        print('\nForming Hessian...')
        t = time.time()
        docc = np.diag(np.ones(self.nocc))
        dvir = np.diag(np.ones(self.nvir))
        eps_diag = self.epsilon[self.nocc:].reshape(-1, 1) - self.epsilon[:self.nocc]

        # Form oNNN MO tensor, oN^4 cost
        MO = np.asarray(self.mints.mo_eri(self.Co, self.C, self.C, self.C))

        H = np.einsum('ai,ij,ab->iajb', eps_diag, docc, dvir)
        H += 4 * MO[:, self.nocc:, :self.nocc, self.nocc:]
        H -= MO[:, self.nocc:, :self.nocc, self.nocc:].swapaxes(0, 2)


        H -= MO[:, :self.nocc, self.nocc:, self.nocc:].swapaxes(1, 2)

        print('...formed Hessian in %.3f seconds.' % (time.time() - t))

        # Invert Hessian (o^3 v^3)
        print('\nInverting Hessian...')
        t = time.time()
        Hinv = np.linalg.inv(H.reshape(self.nocc * self.nvir, -1)).reshape(self.nocc, self.nvir, self.nocc, self.nvir)
        print('...inverted Hessian in %.3f seconds.' % (time.time() - t))

        # Form perturbation response vector for each dipole component
        self.x = []
        for numx in range(3):
            xcomp = np.einsum('iajb,ia->jb', Hinv, self.dipoles_xyz[numx])
            self.x.append(xcomp.reshape(-1))

        self.rhsvecs = []
        for numx in range(3):
            rhsvec = self.dipoles_xyz[numx].reshape(-1)
            self.rhsvecs.append(rhsvec)

    def solve_dynamic_direct(self, omega=0.0):
        # Adapted completely from TDHF.py

        eps_v = self.epsilon[self.nocc:]
        eps_o = self.epsilon[:self.nocc]

        t = time.time()
        I = self.mints.ao_eri()
        v_ijab = np.asarray(self.mints.mo_transform(I, self.Co, self.Co, self.Cv, self.Cv))
        v_iajb = np.asarray(self.mints.mo_transform(I, self.Co, self.Cv, self.Co, self.Cv))
        print('Integral transform took %.3f seconds\n' % (time.time() - t))

        # Since we are time dependent we need to build the full Hessian:
        # | A B |      | D  S | |  x |   |  b |
        # | B A |  - w | S -D | | -x | = | -b |

        # Build A and B blocks
        t = time.time()
        A11  = np.einsum('ab,ij->iajb', np.diag(eps_v), np.diag(np.ones(self.nocc)))
        A11 -= np.einsum('ij,ab->iajb', np.diag(eps_o), np.diag(np.ones(self.nvir)))
        A11 += 2 * v_iajb
        A11 -= v_ijab.swapaxes(1, 2)
        A11 *= 2

        B11  = -2 * v_iajb
        B11 += v_iajb.swapaxes(0, 2)
        B11 *= 2

        # Reshape and jam it together
        nov = self.nocc * self.nvir
        A11.shape = (nov, nov)
        B11.shape = (nov, nov)

        Hess1 = np.hstack((A11, B11))
        Hess2 = np.hstack((B11, A11))
        Hess = np.vstack((Hess1, Hess2))

        S11 = np.zeros_like(A11)
        D11 = np.zeros_like(B11)
        S11[np.diag_indices_from(S11)] = 2

        S1 = np.hstack((S11, D11))
        S2 = np.hstack((D11, -S11))
        S = np.vstack((S1, S2))
        S *= omega
        print('Hessian formation took %.3f seconds\n' % (time.time() - t))

        t = time.time()
        Hinv = np.linalg.inv(Hess - S)
        print('Hessian inversion took %.3f seconds\n' % (time.time() - t))

        self.x = []
        self.rhsvecs = []
        for numx in range(3):
            rhsvec = self.dipoles_xyz[numx].reshape(-1)
            rhsvec = np.concatenate((rhsvec, -rhsvec))
            xcomp = Hinv.dot(rhsvec)
            self.rhsvecs.append(rhsvec)
            self.x.append(xcomp)

    def solve_static_iterative(self, maxiter=20, conv=1.e-9, use_diis=True):

        # Init JK object
        jk = psi4.core.JK.build(self.scf_wfn.basisset())
        jk.initialize()

        # Add blank matrices to the jk object and numpy hooks to C_right
        npC_right = []
        for xyz in range(3):
            jk.C_left_add(self.Co)
            mC = psi4.core.Matrix(self.nbf, self.nocc)
            npC_right.append(np.asarray(mC))
            jk.C_right_add(mC)

        # Build initial guess, previous vectors, diis object, and C_left updates
        self.x = []
        x_old = []
        diis = []
        ia_denom = - self.epsilon[:self.nocc].reshape(-1, 1) + self.epsilon[self.nocc:]
        for xyz in range(3):
            self.x.append(self.dipoles_xyz[xyz] / ia_denom)
            x_old.append(np.zeros(ia_denom.shape))
            diis.append(DIIS_helper())

        # Convert Co and Cv to numpy arrays
        Co = np.asarray(self.Co)
        Cv = np.asarray(self.Cv)

        print('\nStarting CPHF iterations:')
        t = time.time()
        for CPHF_ITER in range(1, maxiter + 1):

            # Update jk's C_right
            for xyz in range(3):
                npC_right[xyz][:] = Cv.dot(self.x[xyz].T)

            # Compute JK objects
            jk.compute()

            # Update amplitudes
            for xyz in range(3):
                # Build J and K objects
                J = np.asarray(jk.J()[xyz])
                K = np.asarray(jk.K()[xyz])

                # Bulid new guess
                X = self.dipoles_xyz[xyz].copy()
                X -= (Co.T).dot(4 * J - K.T - K).dot(Cv)
                X /= ia_denom

                # DIIS for good measure
                if use_diis:
                    diis[xyz].add(X, X - x_old[xyz])
                    X = diis[xyz].extrapolate()
                self.x[xyz] = X.copy()

            # Check for convergence
            rms = []
            for xyz in range(3):
                rms.append(np.max((self.x[xyz] - x_old[xyz]) ** 2))
                x_old[xyz] = self.x[xyz]

            avg_RMS = sum(rms) / 3
            max_RMS = max(rms)

            if max_RMS < conv:
                print('CPHF converged in %d iterations and %.2f seconds.' % (CPHF_ITER, time.time() - t))
                self.rhsvecs = []
                for numx in range(3):
                    rhsvec = self.dipoles_xyz[numx].reshape(-1)
                    self.rhsvecs.append(rhsvec)
                    self.x[numx] = self.x[numx].reshape(-1)
                break

            print('CPHF Iteration %3d: Average RMS = %3.8f  Maximum RMS = %3.8f' %
                  (CPHF_ITER, avg_RMS, max_RMS))

    def solve_dynamic_iterative(self, omega=0.0, maxiter=20, conv=1.e-10, use_diis=True):

        # Init JK object
        jk = psi4.core.JK.build(self.scf_wfn.basisset())
        jk.initialize()

        # Build initial guess, previous vectors, and DIIS objects
        norb = self.scf_wfn.nmo()
        C = np.asarray(self.C)
        rhsmats = [(C.T).dot(np.asarray(dipmat)).dot(C) for dipmat in self.tmp_dipoles]
        x = []
        x_old = []
        diis = []
        ia_denom = self.epsilon[self.nocc:] - self.epsilon[:self.nocc].reshape(-1, 1) - omega
        all_denom = self.epsilon.reshape(-1, 1) - self.epsilon - omega
        for rhsmat in rhsmats:
            U = np.zeros((norb, norb))
            U[:self.nocc, self.nocc:] = rhsmat[:self.nocc, self.nocc:] / ia_denom
            U[self.nocc:, :self.nocc] = rhsmat[self.nocc:, :self.nocc] / -ia_denom.T
            x.append(U)
            x_old.append(np.zeros_like(U))
            diis.append(DIIS_helper())

        # Add blank matrices to the jk object and NumPy hooks to C_right,
        # which will contain MO coefficients contracted with the response
        # matrix. 1 and 2 refer to the first and second terms in the perturbed
        # density build:
        #   P^{x} = C @ U^{x}.T @ C.T - C @ U^{x} @ C.T
        #         = C @ C^{x}_1.T + C^{x}_2 @ C.T
        npCx_1 = []
        npCx_2 = []
        for _ in range(len(rhsmats)):
            mCx_1 = psi4.core.Matrix(self.nbf, self.nocc)
            npCx_1.append(np.asarray(mCx_1))
            jk.C_left_add(self.Co)
            jk.C_right_add(mCx_1)
        for _ in range(len(rhsmats)):
            mCx_2 = psi4.core.Matrix(self.nbf, self.nocc)
            npCx_2.append(np.asarray(mCx_2))
            jk.C_left_add(mCx_2)
            jk.C_right_add(self.Co)

        print('\nStarting CPHF iterations:')
        t = time.time()
        for CPHF_ITER in range(1, maxiter + 1):

            for xyz in range(len(rhsmats)):
                npCx_1[xyz][:] = C.dot(x[xyz].T)[:, :self.nocc]
                npCx_2[xyz][:] = (-C).dot(x[xyz])[:, :self.nocc]

            # Perform generalized J/K build
            jk.compute()
            # Update amplitudes
            for xyz in range(len(rhsmats)):
                # Build J and K objects
                J_1 = np.asarray(jk.J()[xyz])
                K_1 = np.asarray(jk.K()[xyz])
                J_2 = np.asarray(jk.J()[xyz + len(rhsmats)])
                K_2 = np.asarray(jk.K()[xyz + len(rhsmats)])

                # Build new guess: work in the full [norb, norb] space, then
                # select only those parameters corresponding to
                # occ->virt/virt->occ rotation, leaving the occ-occ and
                # virt-virt parameters zero.
                U = x[xyz].copy()
                upd = (rhsmats[xyz] - (x[xyz] * all_denom - (C.T).dot(2 * J_1 - K_1 + 2 * J_2 - K_2).dot(C))) / all_denom
                U[:self.nocc, self.nocc:] += upd[:self.nocc, self.nocc:]
                U[self.nocc:, :self.nocc] += upd[self.nocc:, :self.nocc]
                # DIIS for good measure
                if use_diis:
                    diis[xyz].add(U, U - x_old[xyz])
                    U = diis[xyz].extrapolate()
                x[xyz] = U.copy()

            # Check for convergence
            rms = []
            for xyz in range(3):
                rms.append(np.max((x[xyz] - x_old[xyz]) ** 2))
                x_old[xyz] = x[xyz]

            avg_RMS = sum(rms) / 3
            max_RMS = max(rms)

            if max_RMS < conv:
                print('CPHF converged in %d iterations and %.2f seconds.' % (CPHF_ITER, time.time() - t))
                self.rhsvecs = []
                self.x = []
                for numx in range(3):
                    self.rhsvecs.append(
                        np.concatenate(
                            (
                                self.dipoles_xyz[numx].reshape(-1),
                                -self.dipoles_xyz[numx].T.reshape(-1),
                            )
                        )
                    )
                    self.x.append(
                        np.concatenate(
                            (
                                x[numx][: self.nocc, self.nocc :].reshape(-1),
                                x[numx][self.nocc :, : self.nocc].reshape(-1),
                            )
                        )
                    )
                break

            print('CPHF Iteration %3d: Average RMS = %3.8f  Maximum RMS = %3.8f' %
                  (CPHF_ITER, avg_RMS, max_RMS))

    def form_polarizability(self):
        self.polar = np.empty((3, 3))
        for numx in range(3):
            for numf in range(3):
                self.polar[numx, numf] = self.x[numx].dot(self.rhsvecs[numf])

if __name__ == '__main__':
    print('\n')
    print('@test_CPHF running CPHF.py')

    from CPHF import *

    from helper_CPHF import helper_CPHF

    helper = helper_CPHF(mol)

    print('\n')
    print('@test_CPHF running solve_static_direct')

    helper.solve_static_direct()
    helper.form_polarizability()
    assert np.allclose(polar, helper.polar, rtol=0, atol=1.e-5)

    print('\n')
    print('@test_CPHF running solve_static_iterative')

    helper.solve_static_iterative()
    helper.form_polarizability()
    assert np.allclose(polar, helper.polar, rtol=0, atol=1.e-5)

    f = 0.0

    print('\n')
    print('@test_CPHF running solve_dynamic_direct ({})'.format(f))

    helper.solve_dynamic_direct(omega=f)
    helper.form_polarizability()
    assert np.allclose(polar, helper.polar, rtol=0, atol=1.e-5)

    print('\n')
    print('@test_CPHF running solve_dynamic_iterative ({})'.format(f))

    helper.solve_dynamic_iterative(omega=f)
    helper.form_polarizability()
    assert np.allclose(polar, helper.polar, rtol=0, atol=1.e-5)

    f = 0.0773178
    ref = np.array([
        [8.19440121,  0.00000000,  0.00000000],
        [0.00000000, 12.75967150,  0.00000000],
        [0.00000000,  0.00000000, 10.25213939]
    ])

    print('\n')
    print('@test_CPHF running solve_dynamic_direct ({})'.format(f))

    helper.solve_dynamic_direct(omega=f)
    helper.form_polarizability()
    assert np.allclose(ref, helper.polar, rtol=0, atol=1.e-5)

    print('\n')
    print('@test_CPHF running solve_dynamic_iterative ({})'.format(f))

    helper.solve_dynamic_iterative(omega=f)
    helper.form_polarizability()
    assert np.allclose(ref, helper.polar, rtol=0, atol=1.e-5)
