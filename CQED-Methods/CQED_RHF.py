"""
A restricted Hartree-Fock script using the Psi4NumPy Formalism

References:
- Algorithm taken from [Szabo:1996], pp. 146
- Equations taken from [Szabo:1996]
"""

__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-9-30"

import time
import numpy as np

np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
psi4.set_memory("500 MB")
psi4.core.set_output_file("output.dat", False)

# Memory for numpy in GB
numpy_memory = 2

mol = psi4.geometry(
    """
H
Mg 1 1.5
1 1
"""
)

# mol = psi4.geometry("""
# O
# H 1 1.1
# H 1 1.1 2 104
# symmetry c1
# """)

psi4.set_options({'basis':        'sto-3g',
                  'scf_type':     'pk',
                  'reference':    'rhf',
                  'mp2_type':     'conv',
                  'save_jk': True,
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

# Set defaults
maxiter = 40
E_conv = 1.0e-6
D_conv = 1.0e-3

# electric field parameter
Ex = 0
Ey = 0
Ez = 0
lam = np.array([Ex, Ey, Ez])

# Integral generation from Psi4's MintsHelper
# wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
# t = time.time()
# mints = psi4.core.MintsHelper(wfn.basisset())
scf_e, wfn = psi4.energy("scf", return_wfn=True)

print("finished SCF and here are the vecs!")
print(wfn.Ca())
mints = psi4.core.MintsHelper(wfn.basisset())

# ==> Nuclear Repulsion Energy <==
E_nuc = mol.nuclear_repulsion_energy()

S = np.asarray(mints.ao_overlap())

# Get nbf and ndocc for closed shell molecules
nbf = S.shape[0]
ndocc = wfn.nalpha()

print("\nNumber of occupied orbitals: %d" % ndocc)
print("Number of basis functions: %d" % nbf)

# Run a quick check to make sure everything will fit into memory
I_Size = (nbf ** 4) * 8.0e-9
print("\nSize of the ERI tensor will be %4.2f GB." % I_Size)

# Estimate memory usage
memory_footprint = I_Size * 1.5
if I_Size > numpy_memory:
    psi4.core.clean()
    raise Exception(
        "Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB."
        % (memory_footprint, numpy_memory)
    )

# Compute required quantities for SCF
print("V")
V = np.asarray(mints.ao_potential())
print(V)
T = np.asarray(mints.ao_kinetic())
I = np.asarray(mints.ao_eri())

# Transformation vectors from ordinary SCF
C = np.asarray(wfn.Ca())
print("SCF E")
print(scf_e)
print("C")
print(C)

# number of doubly occupied orbitals
ndocc = wfn.nalpha()

# Extra terms for Pauli-Fierz Hamiltonian
# nuclear dipole
mu_nuc_x = mol.nuclear_dipole()[0]
mu_nuc_y = mol.nuclear_dipole()[1]
mu_nuc_z = mol.nuclear_dipole()[2]

# dipole arrays in AO basis
mu_x_ao = np.asarray(mints.ao_dipole()[0])
mu_y_ao = np.asarray(mints.ao_dipole()[1])
my_z_ao = np.asarray(mints.ao_dipole()[2])

print(mu_x_ao)
# transform dipole array to MO basis from ordinary RHF (no photon)
mu_x = np.dot(C.T, mu_x_ao).dot(C)
mu_y = np.dot(C.T, mu_y_ao).dot(C)
mu_z = np.dot(C.T, mu_z_ao).dot(C)

# compute components of electronic dipole moment <mu> from ordinary RHF (no photon)
mu_exp_x = 0.0
mu_exp_y = 0.0
mu_exp_z = 0.0
for i in range(0, ndocc):
    # double because this is only alpha terms!
    mu_exp_x += 2 * mu_x[i, i]
    mu_exp_y += 2 * mu_y[i, i]
    mu_exp_z += 2 * mu_z[i, i]

# We need to carry around the electric field dotted into the nuclear dipole moment
# and the electric field dotted into the RHF electronic dipole expectation value...
# so let's compute them here!

# lambda . mu_nuc
l_dot_mu_nuc = lam[0] * mu_nuc_x + lam[1] * mu_nuc_y + lam[2] * mu_nuc_z
l_dot_mu_el = lam[0] * mu_exp_x + lam[1] * mu_exp_y + lam[2] * mu_exp_z
l_dot_mu = l_dot_mu_nuc + l_dot_mu_el

# dipole constants to add to E_RHF
l_dot_mu_constant = l_dot_mu_nuc ** 2 + l_dot_mu_nuc * l_dot_mu + l_dot_mu ** 2

# quadrupole arrays
Q_xx = np.asarray(mints.ao_quadrupole()[0])
Q_xy = np.asarray(mints.ao_quadrupole()[1])
Q_xz = np.asarray(mints.ao_quadrupole()[2])
Q_yy = np.asarray(mints.ao_quadrupole()[3])
Q_yz = np.asarray(mints.ao_quadrupole()[4])
Q_zz = np.asarray(mints.ao_quadrupole()[5])

# print('\nTotal time taken for integrals: %.3f seconds.' % (time.time() - t))
t = time.time()

# Build H_core: [Szabo:1996] Eqn. 3.153, pp. 141
# plus terms from Pauli-Fierz Hamiltonian
# ordinary H_core
H_0 = T + V

# Pauli-Fierz 1-e quadrupole terms
Q_PF = lam[0] * lam[0] * Q_xx
Q_PF += lam[1] * lam[1] * Q_yy
Q_PF += lam[2] * lam[2] * Q_zz
Q_PF += 2 * lam[0] * lam[1] * Q_xy
Q_PF += 2 * lam[0] * lam[2] * Q_xz
Q_PF += 2 * lam[1] * lam[2] * Q_yz

# Pauli-Fierz 1-e dipole terms scaled by l . <mu>
d_PF = 2 * l_dot_mu * lam[0] * mu_x
d_PF += 2 * l_dot_mu * lam[1] * mu_y
d_PF += 2 * l_dot_mu * lam[2] * mu_z
H = H_0 + Q_PF + d_PF

# Orthogonalizer A = S^(-1/2) using Psi4's matrix power.
A = mints.ao_overlap()
A.power(-0.5, 1.0e-16)
A = np.asarray(A)

# Calculate initial core guess: [Szabo:1996] pp. 145
Hp = A.dot(H).dot(A)  # Eqn. 3.177
e, C2 = np.linalg.eigh(Hp)  # Solving Eqn. 1.178
C = A.dot(C2)  # Back transform, Eqn. 3.174
Cocc = C[:, :ndocc]

D = np.einsum("pi,qi->pq", Cocc, Cocc)  # [Szabo:1996] Eqn. 3.145, pp. 139

print("\nTotal time taken for setup: %.3f seconds" % (time.time() - t))

print("\nStart SCF iterations:\n")
t = time.time()
E = 0.0
Enuc = mol.nuclear_repulsion_energy()
Eold = 0.0
Dold = np.zeros_like(D)

E_1el = np.einsum("pq,pq->", H + H, D) + Enuc + l_dot_mu_constant
print("One-electron energy = %4.16f" % E_1el)

for SCF_ITER in range(1, maxiter + 1):

    # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141
    J = np.einsum("pqrs,rs->pq", I, D)
    K = np.einsum("prqs,rs->pq", I, D)
    mu_x
    # Pauli-Fierz dipole-dipole matrices
    M_xx = np.einsum("pq,rs,rs->pq", lam[0] * mu_x, lam[0] * mu_x, D)
    M_yy = np.einsum("pq,rs,rs->pq", lam[1] * mu_y, lam[1] * mu_y, D)
    M_zz = np.einsum("pq,rs,rs->pq", lam[2] * mu_z, lam[2] * mu_z, D)

    M_xy = np.einsum("pq,rs,rs->pq", lam[0] * mu_x, lam[1] * mu_y, D)
    M_xz = np.einsum("pq,rs,rs->pq", lam[0] * mu_x, lam[2] * mu_z, D)
    M_yz - np.einsum("pq,rs,rs->pq", lam[1] * mu_y, lam[2] * mu_z, D)

    # Pauli-Fierz dipole-dipole "exchange" terms
    N_xx = np.einsum("pr,qs,rs->pq", lam[0] * mu_x, lam[0] * mu_x, D)
    N_yy = np.einsum("pr,qs,rs->pq", lam[1] * mu_y, lam[1] * mu_y, D)
    N_zz = np.einsum("pr,qs,rs->pq", lam[2] * mu_z, lam[2] * mu_z, D)

    N_xy = np.einsum("pr,qs,rs->pq", lam[0] * mu_x, lam[1] * mu_y, D)
    N_xz = np.einsum("pr,qs,rs->pq", lam[0] * mu_x, lam[2] * mu_z, D)
    N_yz = np.einsum("pr,qs,rs->pq", lam[1] * mu_y, lam[2] * mu_z, D)

    # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141 +
    # Pauli-Fierz terms
    F = H + J * 2 - K
    F += lam[0] ** 2 * M_xx
    F += lam[1] ** 2 * M_yy
    F += lam[2] ** 2 * M_zz

    F += 2 * lam[0] * lam[1] * M_xy
    F += 2 * lam[0] * lam[2] * M_xz
    F += 2 * lam[1] * lam[2] * M_yz

    F -= 0.5 * lam[0] ** 2 * N_xx
    F -= 0.5 * lam[1] ** 2 * N_yy
    F -= 0.5 * lam[2] ** 2 * N_zz

    F -= lam[0] * lam[1] * N_xy
    F -= lam[0] * lam[2] * N_xz
    F -= lam[1] * lam[2] * N_yz

    diis_e = np.einsum("ij,jk,kl->il", F, D, S) - np.einsum("ij,jk,kl->il", S, D, F)
    diis_e = A.dot(diis_e).dot(A)
    dRMS = np.mean(diis_e ** 2) ** 0.5

    # SCF energy and update: [Szabo:1996], Eqn. 3.184, pp. 150
    SCF_E = np.einsum("pq,pq->", F + H, D) + Enuc

    print(
        "SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E"
        % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS)
    )
    if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
        break

    Eold = SCF_E
    Dold = D

    # Diagonalize Fock matrix: [Szabo:1996] pp. 145
    Fp = A.dot(F).dot(A)  # Eqn. 3.177
    e, C2 = np.linalg.eigh(Fp)  # Solving Eqn. 1.178
    C = A.dot(C2)  # Back transform, Eqn. 3.174
    Cocc = C[:, :ndocc]
    D = np.einsum("pi,qi->pq", Cocc, Cocc)  # [Szabo:1996] Eqn. 3.145, pp. 139

    if SCF_ITER == maxiter:
        psi4.core.clean()
        raise Exception("Maximum number of SCF cycles exceeded.")

print("Total time for SCF iterations: %.3f seconds \n" % (time.time() - t))

print("Final SCF energy: %.8f hartree" % SCF_E)
SCF_E_psi = psi4.energy("SCF")
psi4.compare_values(SCF_E_psi, SCF_E, 6, "SCF Energy")
