"""
A Psi4 input script to compute CIS energy from a SCF reference

References:
Algorithms were taken directly from Daniel Crawford's programming website:
http://github.com/CrawfordGroup/ProgrammingProjects
Equations from [Szabo:1996]
"""

__authors__ = "Tianyuan Zhang"
__credits__ = ["Tianyuan Zhang", "Jeffrey B. Schriber", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-05-26"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
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

print('\nStarting SCF and integral build...')
t = time.time()

# First compute SCF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Grab data from wavfunction class
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()
nvirt = nmo - ndocc
nDet_S = ndocc * nvirt * 2

# Compute size of SO-ERI tensor in GB
ERI_Size = (nmo**4) * 128e-9
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

# Update H, transform to MO basis and tile for alpha/beta spin
H = np.einsum('uj,vi,uv', C, C, H)
H = np.repeat(H, 2, axis=0)
H = np.repeat(H, 2, axis=1)

# Make H block diagonal
spin_ind = np.arange(H.shape[0], dtype=np.int) % 2
H *= (spin_ind.reshape(-1, 1) == spin_ind)

print('..finished transformation in %.3f seconds.\n' % (time.time() - t))

from helper_CI import Determinant, HamiltonianGenerator
from itertools import combinations

print('Generating %d CIS singlet Determinants...' % (nDet_S + 1))
t = time.time()

occList = [i for i in range(ndocc)]
det_ref = Determinant(alphaObtList=occList, betaObtList=occList)
detList = det_ref.generateSingleExcitationsOfDet(nmo)
detList.append(det_ref)

print('..finished generating determinants in %.3f seconds.\n' % (time.time() - t))

print('Generating Hamiltonian Matrix...')

t = time.time()
Hamiltonian_generator = HamiltonianGenerator(H, MO)
Hamiltonian_matrix = Hamiltonian_generator.generateMatrix(detList)

print('..finished generating Matrix in %.3f seconds.\n' % (time.time() - t))

print('Diagonalizing Hamiltonian Matrix...')

t = time.time()

e_cis, wavefunctions = np.linalg.eigh(Hamiltonian_matrix)
print('..finished diagonalization in %.3f seconds.\n' % (time.time() - t))

print('# Determinants:     % 16d' % (len(detList)))

print('SCF energy:         % 16.10f' % (scf_e))

hartree2eV = 27.211

print('\nCIS Excitation Energies (Singlets only):')
print(' #        Hartree                  eV')
print('--  --------------------  --------------------')
for i in range(1, len(e_cis)):
    excit_e = e_cis[i] + mol.nuclear_repulsion_energy() - scf_e
    print('%2d %20.10f %20.10f' % (i, excit_e, excit_e * hartree2eV))
