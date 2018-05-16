"""
Implementation of RHF with convergence acceleration via Direct
Inversion of the Iteravite Subspace (DIIS).

References:
- RHF algorithm & equations from [Szabo:1996]
- DIIS algorithm adapted from [Sherrill:1998] & [Pulay:1980:393]
- DIIS equations taken from [Sherrill:1998], [Pulay:1980:393], & [Pulay:1969:197]
"""

__authors__   = "Daniel G. A. Smith"
__credits__   = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-9-30"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat',False)

# Memory for numpy in GB
numpy_memory = 2

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

# Set some options
psi4.set_options({"basis": "cc-pvdz",
                  "scf_type": "pk",
                  "e_convergence": 1e-8})

# Set defaults
maxiter = 40
E_conv = 1.0E-8
D_conv = 1.0E-3

# Integral generation from Psi4's MintsHelper
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
S = np.asarray(mints.ao_overlap())

# Get nbf and ndocc for closed shell molecules
nbf = wfn.nso()
ndocc = wfn.nalpha()

print('\nNumber of occupied orbitals: %d' % ndocc)
print('Number of basis functions: %d' % nbf)

# Run a quick check to make sure everything will fit into memory
I_Size = (nbf**4) * 8.e-9
print("\nSize of the ERI tensor will be %4.2f GB." % I_Size)

# Estimate memory usage
memory_footprint = I_Size * 1.5
if I_Size > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Compute required quantities for SCF
V = np.asarray(mints.ao_potential())
T = np.asarray(mints.ao_kinetic())
I = np.asarray(mints.ao_eri())

print('\nTotal time taken for integrals: %.3f seconds.' % (time.time() - t))

t = time.time()

# Build H_core
H = T + V

# Orthogonalizer A = S^(-1/2)
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

# Calculate initial core guess: [Szabo:1996] pp. 145
Hp = A.dot(H).dot(A)            # Eqn. 3.177
e, C2 = np.linalg.eigh(Hp)      # Solving Eqn. 1.178
C = A.dot(C2)                   # Back transform, Eqn. 3.174
Cocc = C[:, :ndocc]

D = np.einsum('pi,qi->pq', Cocc, Cocc) # [Szabo:1996] Eqn. 3.145, pp. 139

print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))

print('\nStart SCF iterations:\n')
t = time.time()
E = 0.0
Enuc = mol.nuclear_repulsion_energy()
Eold = 0.0

Fock_list = []
DIIS_error = []

for SCF_ITER in range(1, maxiter + 1):

    # Build fock matrix
    J = np.einsum('pqrs,rs->pq', I, D)
    K = np.einsum('prqs,rs->pq', I, D)
    F = H + J * 2 - K

    # DIIS error build w/ HF analytic gradient ([Pulay:1969:197])
    diis_e = np.einsum('ij,jk,kl->il', F, D, S) - np.einsum('ij,jk,kl->il', S, D, F)
    diis_e = A.dot(diis_e).dot(A)
    Fock_list.append(F)
    DIIS_error.append(diis_e)
    dRMS = np.mean(diis_e**2)**0.5

    # SCF energy and update: [Szabo:1996], Eqn. 3.184, pp. 150
    SCF_E = np.einsum('pq,pq->', F + H, D) + Enuc

    print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E'
          % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
    if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
        break

    Eold = SCF_E

    if SCF_ITER >= 2:

        # Limit size of DIIS vector
        diis_count = len(Fock_list)
        if diis_count > 6:
            # Remove oldest vector
            del Fock_list[0]
            del DIIS_error[0]
            diis_count -= 1

        # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
        B = np.empty((diis_count + 1, diis_count + 1))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0
        for num1, e1 in enumerate(DIIS_error):
            for num2, e2 in enumerate(DIIS_error):
                if num2 > num1: continue
                val = np.einsum('ij,ij->', e1, e2)
                B[num1, num2] = val
                B[num2, num1] = val

        # normalize
        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

        # Build residual vector, [Pulay:1980:393], Eqn. 6, RHS
        resid = np.zeros(diis_count + 1)
        resid[-1] = -1

        # Solve Pulay equations, [Pulay:1980:393], Eqn. 6
        ci = np.linalg.solve(B, resid)

        # Calculate new fock matrix as linear
        # combination of previous fock matrices
        F = np.zeros_like(F)
        for num, c in enumerate(ci[:-1]):
            F += c * Fock_list[num]

    # Diagonalize Fock matrix
    Fp = A.dot(F).dot(A)
    e, C2 = np.linalg.eigh(Fp)
    C = A.dot(C2)
    Cocc = C[:, :ndocc]
    D = np.einsum('pi,qi->pq', Cocc, Cocc)

    if SCF_ITER == maxiter:
        psi4.core.clean()
        raise Exception("Maximum number of SCF cycles exceeded.")

print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - t))

print('Final SCF energy: %.8f hartree' % SCF_E)

# Compare to Psi4
SCF_E_psi = psi4.energy('SCF')
psi4.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')
