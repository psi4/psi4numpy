"""
A reference implementation of second-order Moller-Plesset perturbation theory.

References:
- Algorithms and equations were taken directly from Daniel Crawford's programming website:
http://github.com/CrawfordGroup/ProgrammingProjects

Special thanks to Rob Parrish for initial assistance with libmints.
"""

__authors__    = "Daniel G. A. Smith"
__credits__   = ["Daniel G. A. Smith", "Dominic A. Sirianni", "Rob Parrish"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-05-23"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2


mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")


psi4.set_options({'basis': 'aug-cc-pvdz',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

# Check energy against psi4?
check_energy = False

print('\nStarting SCF and integral build...')
t = time.time()

# First compute SCF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Grab data from wavfunction class 
ndocc = wfn.nalpha()
nmo = wfn.nmo()
SCF_E = wfn.energy()
eps = np.asarray(wfn.epsilon_a())

# Compute size of ERI tensor in GB
ERI_Size = (nmo ** 4) * 8e-9
print('Size of the ERI/MO tensor will be %4.2f GB.' % ERI_Size)
memory_footprint = ERI_Size * 2.5
if memory_footprint > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (memory_footprint, numpy_memory))

print('Building MO integrals.')
# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
Co = wfn.Ca_subset("AO", "OCC")
Cv = wfn.Ca_subset("AO", "VIR")
MO = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))

Eocc = eps[:ndocc]
Evirt = eps[ndocc:]

print('Shape of MO integrals: %s' % str(MO.shape))
print('\n...finished SCF and integral build in %.3f seconds.\n' % (time.time() - t))

print('Computing MP2 energy...')
t = time.time()
e_denom = 1 / (Eocc.reshape(-1, 1, 1, 1) - Evirt.reshape(-1, 1, 1) + Eocc.reshape(-1, 1) - Evirt)

# Get the two spin cases
MP2corr_OS = np.einsum('iajb,iajb,iajb->', MO, MO, e_denom)
MP2corr_SS = np.einsum('iajb,iajb,iajb->', MO - MO.swapaxes(1, 3), MO, e_denom)
print('...MP2 energy computed in %.3f seconds.\n' % (time.time() - t))

MP2corr_E = MP2corr_SS + MP2corr_OS
MP2_E = SCF_E + MP2corr_E

SCS_MP2corr_E = MP2corr_SS / 3 + MP2corr_OS * (6. / 5)
SCS_MP2_E = SCF_E + SCS_MP2corr_E

print('MP2 SS correlation energy:         %16.10f' % MP2corr_SS)
print('MP2 OS correlation energy:         %16.10f' % MP2corr_OS)

print('\nMP2 correlation energy:            %16.10f' % MP2corr_E)
print('MP2 total energy:                  %16.10f' % MP2_E)

print('\nSCS-MP2 correlation energy:        %16.10f' % MP2corr_SS)
print('SCS-MP2 total energy:              %16.10f' % SCS_MP2_E)

if check_energy:
    psi4.energy('MP2')
    psi4.compare_values(psi4.core.get_variable('MP2 TOTAL ENERGY'), MP2_E, 6, 'MP2 Energy')
    psi4.compare_values(psi4.core.get_variable('SCS-MP2 TOTAL ENERGY'), SCS_MP2_E, 6, 'SCS-MP2 Energy')


# Natural orbitals as a bonus
lam_menf = MO + (MO - MO.swapaxes(1,3))
amp_ienf = MO * e_denom

# Compute occupied and virtual MP2 densities
Gij = np.einsum('ienf,menf->im', amp_ienf, lam_menf)
Gab = np.einsum('manf,menf->ea', amp_ienf, lam_menf)

# MP2 Density matrix
D_occ = 0.25 * (Gij + Gij.T)
D_occ += np.diag(np.ones(ndocc)) * 2
D_vir = -0.25 * (Gab + Gab.T)

# Build full D and diagonalize
D = np.zeros((nmo, nmo))
D[:ndocc, :ndocc] = D_occ
D[ndocc:, ndocc:] = D_vir

evals, evecs = np.linalg.eigh(D)

# Question for the audience, what should it be?
print("\nThe sum of the natural occupation numbers is %6.4f" % np.sum(evals))



