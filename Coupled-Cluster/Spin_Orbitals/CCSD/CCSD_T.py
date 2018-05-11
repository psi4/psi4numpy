"""
Script to compute the CCSD(T) electronic correlation energy,
from a RHF reference wavefunction.

References:
- Algorithms & equations taken directly from Project #6 of 
Daniel Crawford's programming website:
http://github.com/CrawfordGroup/ProgrammingProjects
"""

__authors__   =  "Daniel G. A. Smith"
__credits__   =  ["Daniel G. A. Smith", "Lori A. Burns"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2014-07-29"

import time
import numpy as np
from helper_CC import *
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

numpy_memory = 2

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis': 'cc-pVDZ'})

# For numpy
compare_psi4 = True

# Compute CCSD
ccsd = helper_CCSD(mol, memory=2)
ccsd.compute_energy()

CCSDcorr_E = ccsd.ccsd_corr_e
CCSD_E = ccsd.ccsd_e

print('\nFinal CCSD correlation energy:          % 16.10f' % CCSDcorr_E)
print('Total CCSD energy:                      % 16.10f' % CCSD_E)

# Triples correction required o^3v^3 storage due the noddy algorithm
T_Size = (ccsd.nocc ** 3 * ccsd.nvirt ** 3) * 8e-9
print("\nSize of the T3 tensor will be %4.2f GB." % T_Size)
memory_footprint = T_Size * 4.2
if memory_footprint > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization for pertubative triples (%4.2f GB) exceeds numpy_memory limit of %4.2f GB." % (memory_footprint, numpy_memory))

# P(i/jk) * P(a/bc)
# (ijk - jik - kji) * (abc - bac - cba)
# (ijkabc - ijkbac - ijkcba - jikabc + jikbac + jikcba - kjiabc + kjicac + kjicba)

# Build disconnected t3
print('\nBuilding disconnected T3...')
t = time.time()
tmp = np.einsum('ia,jkbc->ijkabc', ccsd.t1, ccsd.get_MO('oovv'))
t3d = tmp.copy()
t3d -= np.einsum('ijkabc->ijkbac', tmp)
t3d -= np.einsum('ijkabc->ijkcba', tmp)
t3d -= np.einsum('ijkabc->jikabc', tmp)
t3d += np.einsum('ijkabc->jikbac', tmp)
t3d += np.einsum('ijkabc->jikcba', tmp)
t3d -= np.einsum('ijkabc->kjiabc', tmp)
t3d += np.einsum('ijkabc->kjibac', tmp)
t3d += np.einsum('ijkabc->kjicba', tmp)
print('...built disconnected T3 in %.3f seconds.' % (time.time() - t))

# Build connected t3
print('\nBuilding connected T3...')
t = time.time()
tmp = ndot('jkae,eibc->ijkabc', ccsd.t2, ccsd.get_MO('vovv')).copy()
tmp -= ndot('imbc,majk->ijkabc', ccsd.t2, ccsd.get_MO('ovoo'))
t3c = tmp.copy()
t3c -= np.einsum('ijkabc->ijkbac', tmp)
t3c -= np.einsum('ijkabc->ijkcba', tmp)
t3c -= np.einsum('ijkabc->jikabc', tmp)
t3c += np.einsum('ijkabc->jikbac', tmp)
t3c += np.einsum('ijkabc->jikcba', tmp)
t3c -= np.einsum('ijkabc->kjiabc', tmp)
t3c += np.einsum('ijkabc->kjibac', tmp)
t3c += np.einsum('ijkabc->kjicba', tmp)
print('...built connected T3 in %.3f seconds.' % (time.time() - t))

# Form last intermediate
tmp = t3c + t3d

# Construct D3
Focc = np.diag(ccsd.get_F('oo'))
Fvir = np.diag(ccsd.get_F('vv'))
Dijkabc = Focc.reshape(-1, 1, 1, 1, 1, 1) + Focc.reshape(-1, 1, 1, 1, 1) + Focc.reshape(-1, 1, 1, 1)
Dijkabc = Dijkabc - Fvir.reshape(-1, 1, 1) - Fvir.reshape(-1, 1) - Fvir
tmp /= Dijkabc

# Compute energy expression
Pert_T = (1.0/36) * np.einsum('ijkabc,ijkabc', t3c, tmp)

CCSD_T_E = CCSD_E + Pert_T

print('\nPertubative (T) correlation energy:     % 16.10f' % Pert_T)
print('Total CCSD(T) energy:                   % 16.10f' % CCSD_T_E)
if compare_psi4:
    psi4.compare_values(psi4.energy('CCSD(T)'), CCSD_T_E, 6, 'CCSD(T) Energy')

