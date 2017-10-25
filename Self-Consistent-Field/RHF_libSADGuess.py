"""
A restricted Hartree-Fock script using the Psi4NumPy Formalism
"""

__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-9-30"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
psi4.set_memory('500 MB')
psi4.core.set_output_file("output.dat", False)

# Memory for numpy in GB
numpy_memory = 2

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8})

# Set defaults
maxiter = 40
E_conv = 1.0E-6
D_conv = 1.0E-3

# Initial guess toggle. True for SAD, False for CORE
SAD_Guess = True

# Integral generation from Psi4's MintsHelper
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
S = np.asarray(mints.ao_overlap())

# Get nbf and ndocc for closed shell molecules
nbf = S.shape[0]
ndocc = wfn.nalpha()

print('\nNumber of occupied orbitals: %d' % ndocc)
print('Number of basis functions: %d' % nbf)

# Run a quick check to make sure everything will fit into memory
I_Size = (nbf**4) * 8.e-9
print("\nSize of the ERI tensor will be %4.2f GB." % I_Size)

# Estimate memory usage
memory_footprint = I_Size * 1.5
if I_Size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Compute required quantities for SCF
V = np.asarray(mints.ao_potential())
T = np.asarray(mints.ao_kinetic())
I = np.asarray(mints.ao_eri())

print('\nTotal time taken for integrals: %.3f seconds.' % (time.time() - t))
t = time.time()

# Construct Hamiltonian
H = T + V

# Get guess, specified by SAD_Guess toggle
if(SAD_Guess):
    # Set SAD basis sets
    nbeta = wfn.nbeta()
    psi4.core.prepare_options_for_module("SCF")
    sad_basis_list = psi4.core.BasisSet.build(wfn.molecule(), "ORBITAL",
        psi4.core.get_global_option("BASIS"), puream=wfn.basisset().has_puream(),
                                         return_atomlist=True)

    sad_fitting_list = psi4.core.BasisSet.build(wfn.molecule(), "DF_BASIS_SAD",
        psi4.core.get_option("SCF", "DF_BASIS_SAD"), puream=wfn.basisset().has_puream(),
                                           return_atomlist=True)

    # Use Psi4 SADGuess object to build the SAD Guess
    SAD = psi4.core.SADGuess.build_SAD(wfn.basisset(), sad_basis_list, ndocc, nbeta)
    SAD.set_atomic_fit_bases(sad_fitting_list)
    SAD.compute_guess();
    D = SAD.Da()
else: 
    # Orthogonalizer A = S^(-1/2) using Psi4's matrix power.
    A = mints.ao_overlap()
    A.power(-0.5, 1.e-16)
    A = np.asarray(A)
    
    # Calculate initial core guess
    Hp = A.dot(H).dot(A)
    e, C2 = np.linalg.eigh(Hp)
    C = A.dot(C2)
    Cocc = C[:, :ndocc]
    D = np.einsum('pi,qi->pq', Cocc, Cocc)

# Orthogonalizer A = S^(-1/2) using Psi4's matrix power.
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))

print('\nStart SCF iterations:\n')
t = time.time()
E = 0.0
Enuc = mol.nuclear_repulsion_energy()
Eold = 0.0
Dold = np.zeros_like(D)

for SCF_ITER in range(1, maxiter + 1):

    # Build fock matrix
    J = np.einsum('pqrs,rs->pq', I, D)
    K = np.einsum('prqs,rs->pq', I, D)
    F = H + J * 2 - K

    diis_e = np.einsum('ij,jk,kl->il', F, D, S) - np.einsum('ij,jk,kl->il', S, D, F)
    diis_e = A.dot(diis_e).dot(A)

    # SCF energy and update
    SCF_E = np.einsum('pq,pq->', F + H, D) + Enuc
    dRMS = np.mean(diis_e**2)**0.5

    print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E' % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
    if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
        break

    Eold = SCF_E
    Dold = D

    # Diagonalize Fock matrix
    Fp = A.dot(F).dot(A)
    e, C2 = np.linalg.eigh(Fp)
    C = A.dot(C2)
    Cocc = C[:, :ndocc]
    D = np.einsum('pi,qi->pq', Cocc, Cocc)

    if SCF_ITER == maxiter:
        clean()
        raise Exception("Maximum number of SCF cycles exceeded.")

print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - t))

print('Final SCF energy: %.8f hartree' % SCF_E)
SCF_E_psi = psi4.energy('SCF')
psi4.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')
psi4.compare_values(SCF_ITER, 14, 6, 'SAD Iterations')

