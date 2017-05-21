# A simple Psi 4 input script to compute Full CI from a SCF reference
# Requirements scipy 0.13.0+ and numpy 1.7.2+
#
# Thank Daniel G. A. Smith for coding other projects as reference.
#
# Created by: Tianyuan Zhang
# Date: 5/19/17
# License: GPL v3.0
#

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

compare_psi4 = True

# Memory for Psi4 in GB
# psi4.core.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2


mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")


psi4.set_options({'basis': 'sto-3g',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

# Check energy against psi4?
check_energy = False

print('\nStarting SCF and integral build...')
t = time.time()

# First compute SCF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)
# print scf_e

# Grab data from wavfunction class 
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()

# Compute size of SO-ERI tensor in GB
ERI_Size = (nmo ** 4) * 128e-9
print('\nSize of the SO ERI tensor will be %4.2f GB.' % ERI_Size)
memory_footprint = ERI_Size * 5.2
if memory_footprint > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

print('\nTotal time taken for ERI integrals: %.3f seconds.\n' % (time.time() - t))

#Make spin-orbital MO
print('Starting AO -> spin-orbital MO transformation...')
t = time.time()
MO = np.asarray(mints.mo_spin_eri(C, C))

### Build so Fock matirx

# Update H, transform to MO basis and tile for alpha/beta spin
H = np.einsum('uj,vi,uv', C, C, H)
H = np.repeat(H, 2, axis=0)
H = np.repeat(H, 2, axis=1)

# Make H block diagonal
spin_ind = np.arange(H.shape[0], dtype=np.int) % 2
H *= (spin_ind.reshape(-1, 1) == spin_ind)

print('..finished transformation in %.3f seconds.\n' % (time.time() - t))

from Determinant import Determinant_bits
from MatrixElements import MatrixElements_dense

print('Generating Hamiltonian Matrix...')

t = time.time()
matrix_element = MatrixElements_dense(nmo, ndocc, H, MO)
Hamiltonian = matrix_element.generateMatrix()

print('..finished Hamiltonian Matrix in %.3f seconds.\n' % (time.time() - t))

print('Diagonalizing Hamiltonian Matrix...')

t = time.time()
e_fci, wavefunctions = np.linalg.eigh(Hamiltonian)
fci_mol_e = e_fci[0] + mol.nuclear_repulsion_energy()

print('..finished diagonalization in %.3f seconds.\n' % (time.time() - t))

print('SCF energy:         % 16.10f' % (scf_e))
print('FCI correlation:    % 16.10f' % (fci_mol_e - scf_e))
print('Total FCI energy:   % 16.10f' % (fci_mol_e))

if compare_psi4:
    psi4.driver.p4util.compare_values(psi4.energy('FCI'), fci_mol_e, 6, 'FCI Energy')