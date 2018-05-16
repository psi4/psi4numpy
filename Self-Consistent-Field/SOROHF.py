"""
Restricted open-shell Hartree--Fock (ROHF) using direct second-order
convergence acceleration.

References:
- ROHF equations & algorithms adapted from Psi4
- SO equations & algorithm from [Helgaker:2000]
"""

__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-9-30"

import time
import numpy as np
import helper_HF as scf_helper
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

# Triplet O2
mol = psi4.geometry("""
    0 3
    O
    O 1 1.2
symmetry c1
""")

psi4.set_options({'guess': 'core',
                  'basis': 'aug-cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8,
                  'reference': 'rohf'})

wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))

# Set occupations
nocc = wfn.nalpha()
ndocc = wfn.nbeta()
nsocc = nocc - ndocc

# Set defaults
maxiter = 10
max_micro = 4
micro_print = True
micro_conv = 1.e-3
E_conv = 1.0E-8
D_conv = 1.0E-4

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
S = np.asarray(mints.ao_overlap())
nbf = S.shape[0]

jk = psi4.core.JK.build(wfn.basisset())
jk.initialize()
if nbf > 100:
    raise Exception("This has a N^4 memory overhead, killing if nbf > 100.")

print('\nNumber of doubly occupied orbitals: %d' % ndocc)
print('Number of singly occupied orbitals: %d' % nsocc)
print('Number of basis functions:          %d' % nbf)

V = np.asarray(mints.ao_potential())
T = np.asarray(mints.ao_kinetic())

# Build H_core
H = T + V

# ERI's
I = np.asarray(mints.ao_eri())

# Orthogonalizer A = S^(-1/2)
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

print('\nTotal time taken for integrals: %.3f seconds.' % (time.time() - t))

t = time.time()


def transform(I, C1, C2, C3, C4):
    """
    Transforms the 4-index ERI I with the 4 transformation matrices C1 to C4.
    """
    nao = I.shape[0]
    MO = np.dot(C1.T, I.reshape(nao, -1)).reshape(C1.shape[1], nao, nao, nao)

    MO = np.einsum('qB,Aqrs->ABrs', C2, MO)
    MO = np.einsum('rC,ABrs->ABCs', C3, MO)
    MO = np.einsum('sD,ABCs->ABCD', C4, MO)
    return MO


# Build initial orbitals and density matrices
Hp = A.dot(H).dot(A)
e, Ct = np.linalg.eigh(Hp)
C = A.dot(Ct)
Cnocc = C[:, :nocc]
Docc = np.dot(Cnocc, Cnocc.T)
Cndocc = C[:, :ndocc]
Ddocc = np.dot(Cndocc, Cndocc.T)

t = time.time()
E = 0.0
Enuc = mol.nuclear_repulsion_energy()
Eold = 0.0
iter_type = 'CORE'

# Build a DIIS helper object
diis = scf_helper.DIIS_helper()

print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))

print('\nStart SCF iterations:\n')
t = time.time()

for SCF_ITER in range(1, maxiter + 1):

    # Build a and b fock matrices
    Ja = np.einsum('pqrs,rs->pq', I, Docc)
    Ka = np.einsum('prqs,rs->pq', I, Docc)
    Jb = np.einsum('pqrs,rs->pq', I, Ddocc)
    Kb = np.einsum('prqs,rs->pq', I, Ddocc)
    J = Ja + Jb
    Fa = H + J - Ka
    Fb = H + J - Kb

    # Build MO Fock matrix
    moFa = (C.T).dot(Fa).dot(C)
    moFb = (C.T).dot(Fb).dot(C)

    # Special note on the ROHF Fock matrix (taken from psi4)
    # Fo = open-shell fock matrix = 0.5 Fa
    # Fc = closed-shell fock matrix = 0.5 (Fa + Fb)
    #
    # The effective Fock matrix has the following structure
    #          |  closed     open    virtual
    #  ----------------------------------------
    #  closed  |    Fc     2(Fc-Fo)    Fc
    #  open    | 2(Fc-Fo)     Fc      2Fo
    #  virtual |    Fc       2Fo       Fc

    moFeff = 0.5 * (moFa + moFb)
    moFeff[:ndocc, ndocc:nocc] = moFb[:ndocc, ndocc:nocc]
    moFeff[ndocc:nocc, :ndocc] = moFb[ndocc:nocc, :ndocc]
    moFeff[ndocc:nocc, nocc:] = moFa[ndocc:nocc, nocc:]
    moFeff[nocc:, ndocc:nocc] = moFa[nocc:, ndocc:nocc]

    # Back transform to AO Fock
    Feff = (Ct).dot(moFeff).dot(Ct.T)

    # Build gradient
    IFock = moFeff[:nocc, ndocc:].copy()
    IFock[ndocc:, :nsocc] = 0.0
    diis_e = (Ct[:, :nocc]).dot(IFock).dot(Ct[:, ndocc:].T)
    diis.add(Feff, diis_e)

    # SCF energy and update
    SCF_E = np.einsum('pq,pq->', Docc + Ddocc, H)
    SCF_E += np.einsum('pq,pq->', Docc, Fa)
    SCF_E += np.einsum('pq,pq->', Ddocc, Fb)
    SCF_E *= 0.5
    SCF_E += Enuc

    dRMS = np.mean(diis_e**2)**0.5
    print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E   %s' % \
            (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS, iter_type))
    if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
        break

    #if SCF_ITER == maxiter:
    #    clean()
    #    raise Exception("Maximum number of SCF cycles exceeded.")

    ediff = abs(SCF_E - Eold)
    Eold = SCF_E

    gradient = -4 * IFock.copy()
    gradient[ndocc:] /= 2
    gradient[:, :nsocc] /= 2
    gradient[ndocc:, :nsocc] = 0.0
    grad_dot = np.linalg.norm(gradient)

    #if True:
    if (np.max(np.abs(gradient)) > 0.1):
        # Conventional update
        Feff = diis.extrapolate()
        e, Ct = np.linalg.eigh(Feff)
        C = A.dot(Ct)
        iter_type = 'DIIS'

    else:
        # Second-order update
        Cocc = C[:, :nocc]
        Cvir = C[:, ndocc:]
        nvir = nbf - ndocc

        # Build an approximate ROHF guess
        eps = np.diag(moFeff)
        precon = -4 * (eps[:nocc].reshape(-1, 1) - eps[ndocc:])
        precon[ndocc:, :nsocc] = 1
        precon[ndocc:] /= 2
        guess_x = gradient / precon

        # Start Hessian
        MOovov = transform(I, Cocc, Cvir, Cocc, Cvir)
        MOoovv = transform(I, Cocc, Cocc, Cvir, Cvir)

        IAJB = MOovov.copy()

        IAJB -= 0.5 * np.einsum('pqrs->psrq', MOovov)
        IAJB -= 0.5 * np.einsum('pqrs->qspr', MOoovv)

        iajb = IAJB.copy()

        IAJB += 0.5 * np.einsum('IJ,AB->IAJB', np.diag(np.ones(nocc)), moFa[ndocc:, ndocc:])
        IAJB -= 0.5 * np.einsum('AB,IJ->IAJB', np.diag(np.ones(nvir)), moFa[:nocc, :nocc])

        # We need to zero out the redundant rotations
        IAJB[:, :nsocc, :, :] = 0.0
        IAJB[:, :, :, :nsocc] = 0.0

        iajb += 0.5 * np.einsum('IJ,AB->IAJB', np.diag(np.ones(nocc)), moFb[ndocc:, ndocc:])
        iajb -= 0.5 * np.einsum('AB,IJ->IAJB', np.diag(np.ones(nvir)), moFb[:nocc, :nocc])

        # We need to zero out the redundant rotations
        iajb[:, :, ndocc:, :] = 0.0
        iajb[ndocc:, :, :, :] = 0.0

        IAjb = MOovov.copy()
        for i in range(nsocc):
            IAjb[ndocc + i, :, :, i] += 0.5 * moFb[ndocc:, :nocc]

        # We need to zero out the redundant rotations
        IAjb[:, :, ndocc:, :] = 0.0
        IAjb[:, :nsocc, :, :] = 0.0

        iaJB = np.einsum('pqrs->rspq', IAjb)

        # Build and find x
        Hess = IAJB + IAjb + iaJB + iajb

        Hess *= 4
        ndim = Hess.shape[0] * Hess.shape[1]

        Hess = Hess.reshape(gradient.size, -1)  # Make the hessian square
        Hess[np.diag_indices_from(Hess)] += 1.e-14  # Prevent singularities
        x = np.linalg.solve(Hess, gradient.ravel()).reshape(nocc, nvir)

        # Special orbital rotation, some overlap in the middle
        U = np.zeros((C.shape[1], C.shape[1]))
        U[:nocc, ndocc:] = x
        U[ndocc:, :nocc] = -x.T

        U += 0.5 * np.dot(U, U)
        U[np.diag_indices_from(U)] += 1

        # Easy acess to shmidt orthogonalization
        U, r = np.linalg.qr(U.T)
        #print U

        # Rotate and set orbitals
        Ct = Ct.dot(U)
        C = A.dot(Ct)

        iter_type = 'SOSCF'

    Cnocc = C[:, :nocc]
    Docc = np.dot(Cnocc, Cnocc.T)
    Cndocc = C[:, :ndocc]
    Ddocc = np.dot(Cndocc, Cndocc.T)

print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - t))

print('Final SCF energy: %.8f hartree' % SCF_E)

# Compare to Psi4
SCF_E_psi = psi4.energy('SCF')
psi4.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')
