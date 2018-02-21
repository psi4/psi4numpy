"""
A Psi4 input script to compute excitation energies using EOM CCSD.
This implementation shows how to use the Davidson method to partially
diagonalize the (large) effective Hamiltonian matrix in order to find the lowest
few excited states in the context of the EOM_CCSD method.

A description of the Davidson algorithm can be found in Daniel Crawford's
programming projects (#13)
https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2313

Spin orbital sigma equations can be found in I. Shavitt and R. J. Bartlett,
"Many-Body Methods in Chemistry and Physics: MBPT and Coupled-Cluster Theory".
Cambridge University Press, 2009.

Special thanks to Ashutosh Kumar for Hbar components and help with spin
adaptation.
"""

__authors__ = "Andrew M. James"
__credits__ = ["T. Daniel Crawford", "Ashutosh Kumar", "Andrew M. James"]
__copyright__ = "(c) 2017, The Psi4Numpy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-1-17"

from helper_ccenergy import *
from helper_cchbar import *
from helper_cclambda import *
from helper_cceom import *
import psi4
import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, threshold=200, suppress=True)

# Psi4 setup
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
noreorient
symmetry c1
""")

# EOM options
compare_psi4 = True
nroot = 3
nvec_per_root = 30
e_tol = 1.0e-6
max_iter = 80

# roots per irrep must be set to do the eom calculation with psi4
psi4.set_options({'basis': 'cc-pvdz', 'roots_per_irrep': [nroot]})

# Compute CCSD energy for required integrals and T-amplitudes
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
ccsd = HelperCCEnergy(mol, rhf_e, rhf_wfn)
ccsd.compute_energy()

ccsd_cor_e = ccsd.ccsd_corr_e
ccsd_tot_e = ccsd.ccsd_e
print("\n")
print("{}{}".format('Final CCSD correlation energy:'.ljust(30, '.'),
                    "{:16.15f}".format(ccsd_cor_e).rjust(20, '.')))
print("{}{}".format('Total CCSD energy:'.ljust(30, '.'),
                    "{:16.15f}".format(ccsd_tot_e).rjust(20, ',')))

cchbar = HelperCCHbar(ccsd)

# build CIS guess
ndocc = ccsd.ndocc
nvir = ccsd.nvirt
nov = ndocc * nvir
noovv = nov * nov
print("ndocc = {}".format(ndocc))
print("nvir = {}".format(nvir))
print("nov = {}".format(nov))

hbar_dim = nov + noovv
# L is the dimension of the guess space, we start with nroot*2 guesses
L = nroot * 2

# When L exceeds Lmax we will collapse the guess space so our sub-space
# diagonalization problem does not grow too large
Lmax = nroot * nvec_per_root

# An array to hold the excitation energies
theta = [0.0] * L

# build a HelperCCEom object
cceom = HelperCCEom(ccsd, cchbar)

# Get the approximate diagonal of Hbar
D = np.hstack((cceom.Dia.flatten(), cceom.Dijab.flatten()))

# We build a guess by selecting the lowest values of the approximate diagonal
# in the singles space one for each nroot*2 guesses we want,
# and we insert a guess vector with a 1 in the position corresponding to
# that single excitation and zeros everywhere else.
# This is a decent guess, more complicated one such as using CIS
# eigenvectors are more common in production level codes.
B_idx = D[:nov].argsort()[:nroot * 2]
B = np.eye(hbar_dim)[:, B_idx]

conv = False
eom_start = time.time()
# This is the start of the Davidson iterations
for EOMCCSD_iter in range(0, max_iter + 1):
    # QR decomposition ensures that the columns of B are orthogonal
    B, R = np.linalg.qr(B)
    L = B.shape[1]
    theta_old = theta[:nroot]
    print("EOMCCSD: Iter # {:>6} L = {}".format(EOMCCSD_iter, L))
    # Build up the matrix S, holding the products Hbar*B, aka sigma vectors
    S = np.zeros_like(B)
    for i in range(L):
        B1 = B[:nov, i].reshape(ndocc, nvir).copy()
        B2 = B[nov:, i].reshape(ndocc, ndocc, nvir, nvir).copy()
        S1 = cceom.build_sigma1(B1, B2)
        S2 = cceom.build_sigma2(B1, B2)
        S[:nov, i] += S1.flatten()
        S[nov:, i] += S2.flatten()
    # Build the subspace Hamiltonian
    G = np.dot(B.T, S)
    # Diagonalize it, and sort the eigenvector/eigenvalue pairs
    theta, alpha = np.linalg.eig(G)
    idx = theta.argsort()[:nroot]
    theta = theta[idx]
    alpha = alpha[:, idx]
    # This vector will hold the new guess vectors to add to our space
    add_B = []

    for j in range(nroot):
        # Compute a residual vector "w" for each root we seek
        # Note: for a more robust convergence criteria you can also check
        # that the norm of the residual vector is below some threshold.
        w = np.dot(S, alpha[:, j]) - theta[j] * np.dot(B, alpha[:, j])
        # Precondition the residual vector to form a correction vector
        q = w / (theta[j] - D[j])
        # The correction vectors are added to the set of guesses after each
        # iterations, so L the guess space dimension grows by nroot at each
        # iteration
        add_B.append(q)
        de = abs(theta[j] - theta_old[j])
        print("    Root {}: e = {:>20.12f} de = {:>20.12f} "
              "|r| = {:>20.12f}".format(j, theta[j], de, np.linalg.norm(w)))

    # check convergence
    e_norm = np.linalg.norm(theta[:nroot] - theta_old)
    if (e_norm < e_tol):
        # If converged exit
        conv = True
        break

    else:
        # if we are not converged
        # check the subspace dimension, if it has grown too large, collapse
        # to nroot guesses using the current estimate of the eigenvectors
        if L >= Lmax:
            B = np.dot(B, alpha)
            # These vectors will give the same eigenvalues at the next
            # iteration so to avoid a false convergence we reset the theta
            # vector to theta_old
            theta = theta_old
        else:
            # if not we add the preconditioned residuals to the guess
            # space, and continue. Note that the set will be orthogonalized
            # at the start of the next iteration
            Btup = tuple(B[:, i] for i in range(L)) + tuple(add_B)
            B = np.column_stack(Btup)
if conv:
    print("EOMCCSD Davidson iterations finished in {}s".format(
        time.time() - eom_start))
    print("Davidson Converged!")
    print("Excitation Energies")
    print("{:>6}  {:^20}  {:^20}".format("Root #", "Hartree", "eV"))
    print("{:>6}  {:^20}  {:^20}".format("-" * 6, "-" * 20, "-" * 20))
    for i in range(nroot):
        print("{:>6}  {:>20.12f}  {:>20.12f}".format(i, theta[i],
                                                     theta[i] * 22.211))
else:
    psi4.core.clean()
    raise Exception("EOMCCSD Failed -- Iterations exceeded")

# if requested compare values with psi4
if compare_psi4:
    print("Checking against psi4....")
    psi4.energy('eom-ccsd')
    for i in range(nroot):
        var_str = "CC ROOT {} CORRELATION ENERGY".format(i + 1)
        e_ex = psi4.core.get_variable(var_str)
        psi4.compare_values(e_ex, theta[i], 5, var_str)
