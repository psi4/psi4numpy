"""
A iterative second-order restricted open-shell Hartree-Fock script using the Psi4NumPy Formalism
"""

__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-9-30"

import time
import numpy as np
import helper_HF as scf_helper
import scipy.linalg as SLA
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

# Triplet O2
mol = psi4.geometry("""
    0 5
    O
    O 1 1.2
symmetry c1
""")

psi4.set_options({'guess': 'core',
                  'basis': 'aug-cc-pvtz',
                  'scf_type': 'df',
                  'e_convergence': 1e-8,
                  'reference': 'rohf'})

wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))

# Set occupations
nocc = wfn.nalpha()
ndocc = wfn.nbeta()
nsocc = nocc - ndocc

# Set defaults
maxiter = 15
max_micro = 5
micro_print = True
micro_conv = 5.e-3
E_conv = 1.0E-8
D_conv = 1.0E-8

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
S = np.asarray(mints.ao_overlap())
nbf = S.shape[0]

#I = np.array(mints.ao_eri())

print('\nNumber of doubly occupied orbitals: %d' % ndocc)
print('Number of singly occupied orbitals: %d' % nsocc)
print('Number of basis functions:          %d' % nbf)

V = np.asarray(mints.ao_potential())
T = np.asarray(mints.ao_kinetic())

print('\nTotal time taken for integrals: %.3f seconds.' % (time.time()-t))

t = time.time()

# Build H_core
H = T + V

# Orthogonalizer A = S^(-1/2)
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

def SCF_Hx(x, moFa, moFb, C):
    """
    Compute a hessian vector guess where x is a ov matrix of nonredundant operators.
    """

    Co_a = C[:, :nocc]
    Co_b = C[:, :ndocc]
    C_right_a = np.dot(C[:, nocc:], x[:, nsocc:].T)
    C_right_b = np.dot(C[:, ndocc:], x[:ndocc, :].T)
    J, K = scf_helper.compute_jk(jk, [Co_a, Co_b], [C_right_a, C_right_b])
    J1, J2 = J
    K1, K2 = K

    IAJB = (C[:, :nocc].T).dot(J1 - 0.5 * K1 - 0.5 * K1.T).dot(C[:, ndocc:])
    IAJB += 0.5 * np.dot(x[:, nsocc:], moFa[nocc:, ndocc:])
    IAJB -= 0.5 * np.dot(moFa[:nocc, :nocc], x)
    IAJB[:, :nsocc] = 0.0

    iajb = (C[:, :nocc].T).dot(J2 - 0.5 * K2 - 0.5 * K2.T).dot(C[:, ndocc:])
    iajb += 0.5 * np.dot(x, moFb[ndocc:, ndocc:])
    iajb -= 0.5 * np.dot(moFb[:nocc, :ndocc], x[:ndocc, :])
    iajb[ndocc:, :] = 0.0

    IAjb = (C[:, :nocc].T).dot(J2).dot(C[:, ndocc:])
    IAjb[ndocc:] += 0.5 * np.dot(x[:, :nsocc].T, moFb[:nocc, ndocc:])
    IAjb[:, :nsocc] = 0.0

    iaJB = (C[:, :nocc].T).dot(J1).dot(C[:, ndocc:])
    iaJB[:, :nsocc] += 0.5 * np.dot(moFb[:nocc, nocc:], x[ndocc:, nsocc:].T)
    iaJB[ndocc:] = 0.0

    ret = 4 * (IAJB + IAjb + iaJB + iajb)

    return ret

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

# Initialize the JK object
jk = psi4.core.JK.build(wfn.basisset())
jk.initialize()

# Build a DIIS helper object
diis = scf_helper.DIIS_helper()

print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))

print('\nStart SCF iterations:\n')
t = time.time()

for SCF_ITER in range(1, maxiter + 1):

    # Build a and b fock matrices
    J, K = scf_helper.compute_jk(jk, [C[:, :nocc], C[:, :ndocc]])
    J = J[0] + J[1]
    Fa = H + J - K[0]
    Fb = H + J - K[1]

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
    SCF_E  = np.einsum('pq,pq->', Docc + Ddocc, H)
    SCF_E += np.einsum('pq,pq->', Docc, Fa)
    SCF_E += np.einsum('pq,pq->', Ddocc, Fb)
    SCF_E *= 0.5
    SCF_E += Enuc

    dRMS = np.mean(diis_e**2)**0.5
    print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E   %s' % \
            (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS, iter_type))
    if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
        break

    if SCF_ITER == maxiter:
        psi4.core.clean()
        raise Exception("Maximum number of SCF cycles exceeded.")

    ediff = abs(SCF_E - Eold)
    Eold = SCF_E


    gradient = -4 * IFock.copy()
    gradient[ndocc:] /= 2
    gradient[:, :nsocc] /= 2
    grad_dot = np.vdot(gradient, gradient)

    if (np.max(np.abs(gradient)) > 0.2):
        # Conventional update
        Feff = diis.extrapolate()
        e, Ct = np.linalg.eigh(Feff)
        C = A.dot(Ct)
        iter_type = 'DIIS'

    else:
        # Second-order update
        eps = np.diag(moFeff)
        precon = -3.5 * (eps[:nocc].reshape(-1, 1) - eps[ndocc:])
        precon[ndocc:] *= 0.5
        precon[:, :nsocc] *= 0.5
        precon[ndocc:, :nsocc] = 1
        x = gradient / precon

        Ax = SCF_Hx(x, moFa, moFb, C)
        r = gradient - Ax
        z = r / precon
        p = z.copy()
        rms = (np.vdot(r,r) / grad_dot) ** 0.5
        if micro_print:
            print('Micro Iteration Guess: Rel. RMS = %1.5e' %  (rms))

        # CG iterations
        for rot_iter in range(max_micro):
            rz_old = np.vdot(r, z)

            Ap = SCF_Hx(p, moFa, moFb, C)
            alpha = rz_old / np.vdot(Ap, p)

            x += alpha * p
            r -= alpha * Ap
            z = r / precon

            rms = (np.vdot(r, r) / grad_dot) ** 0.5

            if micro_print:
                print('Micro Iteration %5d: Rel. RMS = %1.5e' %  (rot_iter + 1, rms))
            if rms < micro_conv:
                break

            beta = np.vdot(r, z) / rz_old
            p = z + beta * p

        # Special orbital rotation, some overlap in the middle
        U = np.zeros((C.shape[1], C.shape[1]))
        U[:nocc, ndocc:] = x
        U[ndocc:, :nocc] = -x.T

        U = SLA.expm(U.T)
        # Rotate and set orbitals
        Ct = Ct.dot(U)
        C = C.dot(U)

        iter_type = 'SOSCF, nmicro ' + str(rot_iter + 1)

    Cnocc = C[:, :nocc]
    Docc = np.dot(Cnocc, Cnocc.T)
    Cndocc = C[:, :ndocc]
    Ddocc = np.dot(Cndocc, Cndocc.T)


print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - t))

print('Final SCF energy: %.8f hartree' % SCF_E)

# Compare to Psi4
SCF_E_psi = psi4.energy('SCF')
psi4.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')
