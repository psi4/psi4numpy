"""
Unrestricted Hartree--Fock script using iterative second-order
convergence acceleration via preconditioned conjugate gradients (PCG).

References:
- UHF equations & algorithms from [Szabo:1996]
- SO equations from [Helgaker:2000]
- PCG equations & algorithm from [Shewchuk:1994]
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

# Triplet O2, actually very multireference
mol = psi4.geometry("""
    0 3
    O
    O 1 1.2
symmetry c1
""")

psi4.set_options({'basis': 'aug-cc-pvdz',
                  'scf_type': 'df',
                  'e_convergence': 1e-8,
                  'reference': 'uhf'})

# Set defaults
maxiter = 10
E_conv = 1.0E-8
D_conv = 1.0E-5
max_micro = 4
micro_conv = 5.e-2
micro_print = True

# Integral generation from Psi4's MintsHelper
t = time.time()
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
mints = psi4.core.MintsHelper(wfn.basisset())
S = np.asarray(mints.ao_overlap())

# Occupations
nbf = wfn.nso()
nalpha = wfn.nalpha()
nbeta = wfn.nbeta()

print('\nNumber of doubly occupied orbitals: %d' % nalpha)
print('\nNumber of singly occupied orbitals: %d' % (nalpha - nbeta))
print('Number of basis functions: %d' % nbf)

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


def diag_H(H, nocc):
    Hp = A.dot(H).dot(A)
    e, C2 = np.linalg.eigh(Hp)
    C = A.dot(C2)
    Cocc = C[:, :nocc]
    D = np.einsum('pi,qi->pq', Cocc, Cocc)
    return (C, D)


def SCF_Hx(xa, xb, moFa, Co_a, Cv_a, moFb, Co_b, Cv_b):
    """
    Compute the "matrix-vector" product between electronic Hessian (rank-4) and
    matrix of nonredundant orbital rotations (rank-2).

    Parameters
    ----------
    x : numpy.array
        Matrix of nonredundant rotations.
    moF : numpy.array
        MO-basis Fock matrix
    Co : numpy.array
        Matrix of occupied orbital coefficients.
    Cv : numpy.array
        Matrix of virtual orbital coefficients.

    Returns
    -------
    F : numpy.array
        Hessian product tensor
    """
    Hx_a = np.dot(moFa[:nbeta, :nbeta], xa)
    Hx_a -= np.dot(xa, moFa[nbeta:, nbeta:])

    Hx_b = np.dot(moFb[:nalpha, :nalpha], xb)
    Hx_b -= np.dot(xb, moFb[nalpha:, nalpha:])

    # Build two electron part, M = -4 (4 G_{mnip} - g_{mpin} - g_{npim}) K_{ip}
    # From [Helgaker:2000] Eqn. 10.8.65
    C_right_a = np.einsum('ia,sa->si', -xa, Cv_a)
    C_right_b = np.einsum('ia,sa->si', -xb, Cv_b)

    J, K = scf_helper.compute_jk(jk, [Co_a, Co_b], [C_right_a, C_right_b])

    Jab = J[0] + J[1]
    Hx_a += (Co_a.T).dot(2 * Jab - K[0].T - K[0]).dot(Cv_a)
    Hx_b += (Co_b.T).dot(2 * Jab - K[1].T - K[1]).dot(Cv_b)

    Hx_a *= -4
    Hx_b *= -4

    return (Hx_a, Hx_b)

Ca, Da = diag_H(H, nbeta)
Cb, Db = diag_H(H, nalpha)

t = time.time()
E = 0.0
Enuc = mol.nuclear_repulsion_energy()
Eold = 0.0

# Initialize the JK object
jk = psi4.core.JK.build(wfn.basisset())
jk.initialize()

# Build a DIIS helper object
diisa = scf_helper.DIIS_helper()
diisb = scf_helper.DIIS_helper()

print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))

print('\nStart SCF iterations:\n')
t = time.time()

for SCF_ITER in range(1, maxiter + 1):

    # Build Fock matrices
    J, K = scf_helper.compute_jk(jk, [Ca[:, :nbeta], Cb[:, :nalpha]])
    J = J[0] + J[1]
    Fa = H + J - K[0]
    Fb = H + J - K[1]

    # DIIS error build and update
    diisa_e = Fa.dot(Da).dot(S) - S.dot(Da).dot(Fa)
    diisa_e = (A.T).dot(diisa_e).dot(A)
    diisa.add(Fa, diisa_e)

    diisb_e = Fb.dot(Db).dot(S) - S.dot(Db).dot(Fb)
    diisb_e = (A.T).dot(diisb_e).dot(A)
    diisb.add(Fb, diisb_e)

    # SCF energy and update
    SCF_E  = np.einsum('pq,pq->', Da + Db, H)
    SCF_E += np.einsum('pq,pq->', Da, Fa)
    SCF_E += np.einsum('pq,pq->', Db, Fb)
    SCF_E *= 0.5
    SCF_E += Enuc

    dRMS = 0.5 * (np.mean(diisa_e**2)**0.5 + np.mean(diisb_e**2)**0.5)
    print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E'
          % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
    if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
        break

    Eold = SCF_E

    Co_a = Ca[:, :nbeta]
    Cv_a = Ca[:, nbeta:]
    moF_a = np.dot(Ca.T, Fa).dot(Ca)
    gradient_a = -4 * moF_a[:nbeta, nbeta:]
    gradient_norm_a = np.linalg.norm(gradient_a)

    Co_b = Cb[:, :nalpha]
    Cv_b = Cb[:, nalpha:]
    moF_b = np.dot(Cb.T, Fb).dot(Cb)
    gradient_b = -4 * moF_b[:nalpha, nalpha:]
    gradient_norm_b = np.linalg.norm(gradient_b)

    gradient_norm = gradient_norm_a + gradient_norm_b

    # Conventional updates
    if np.any(np.abs(gradient_a) > 0.3) or np.any(np.abs(gradient_b) > 0.3):
        Fa = diisa.extrapolate()
        Fb = diisb.extrapolate()

        # Diagonalize Fock matrix
        Ca, Da = diag_H(Fa, nbeta)
        Cb, Db = diag_H(Fb, nalpha)

    else:
        so_diis = scf_helper.DIIS_helper()

        # Initial guess & Jacobi preconditioner for alpha & beta
        eps_a = np.diag(moF_a)
        precon_a = -4 * (eps_a[:nbeta].reshape(-1, 1) - eps_a[nbeta:])
        x_a = gradient_a / precon_a

        eps_b = np.diag(moF_b)
        precon_b = -4 * (eps_b[:nalpha].reshape(-1, 1) - eps_b[nalpha:])
        x_b = gradient_b / precon_b

        Hx_a, Hx_b = SCF_Hx(x_a, x_b, moF_a, Co_a, Cv_a, moF_b, Co_b, Cv_b)

        r_a = gradient_a - Hx_a
        z_a = r_a / precon_a
        p_a = z_a.copy()

        r_b = gradient_b - Hx_b
        z_b = r_b / precon_b
        p_b = z_b.copy()

        # PCG Iterations for alpha & beta
        for rot_iter in range(max_micro):
            rz_old = np.vdot(r_a, z_a) + np.vdot(r_b, z_b)

            Hx_a, Hx_b = SCF_Hx(p_a, p_b, moF_a, Co_a, Cv_a, moF_b, Co_b, Cv_b)

            alpha = rz_old / (np.vdot(Hx_a, p_a) + np.vdot(Hx_b, p_b))

            # CG update
            x_a += alpha * p_a
            r_a -= alpha * Hx_a
            z_a = r_a / precon_a

            x_b += alpha * p_b
            r_b -= alpha * Hx_b
            z_b = r_b / precon_b

            x_diis = np.hstack((x_a.ravel(), x_b.ravel()))
            r_diis = np.hstack((r_a.ravel(), r_b.ravel()))
            so_diis.add(x_diis, r_diis)

            rms_a = (np.linalg.norm(r_a) / gradient_norm_a) ** 0.5
            rms_b = (np.linalg.norm(r_b) / gradient_norm_b) ** 0.5

            if gradient_norm > 1.e-2:
                denom = gradient_norm
            else:
                denom = 1.e-2
            rms = ((np.linalg.norm(r_a) + np.linalg.norm(r_b)) / denom) ** 0.5

            if micro_print:
                print('Micro Iteration %2d: Rel. RMS = %1.5e (a: %1.2e, b: %1.2e)' %  (rot_iter + 1, rms, rms_a, rms_b))
            if rms < micro_conv:
                break

            beta = (np.vdot(r_a, z_a) + np.vdot(r_b, z_b)) / rz_old

            p_a = z_a + beta * p_a
            p_b = z_b + beta * p_b

        x = so_diis.extrapolate()
        x_a = x[:x_a.size].reshape(x_a.shape)
        x_b = x[x_a.size:].reshape(x_b.shape)

        # Diagonalize Fock matrix
        Ca, Da = scf_helper.rotate_orbitals(Ca, x_a, True)
        Cb, Db = scf_helper.rotate_orbitals(Cb, x_b, True)

    if SCF_ITER == maxiter:
        psi4.core.clean()
        raise Exception("Maximum number of SCF cycles exceeded.")

print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - t))

spin_mat = (Cb[:, :nalpha].T).dot(S).dot(Ca[:, :nbeta])
spin_contam = min(nbeta, nalpha) - np.vdot(spin_mat, spin_mat)
print('Spin Contamination Metric: %1.5E\n' % spin_contam)

print('Final SCF energy: %.8f hartree' % SCF_E)

# Compare to Psi4
SCF_E_psi = psi4.energy('SCF')
psi4.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')
