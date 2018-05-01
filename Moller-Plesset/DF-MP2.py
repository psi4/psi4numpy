"""
A reference implementation of density-fitted MP2 from a RHF reference.

References: 
Algorithm modified from Rob Parrish's most excellent Psi4 plugin example
Bottom of the page: http://www.psicode.org/developers.php
"""

__authors__   = "Daniel G. A. Smith"
__credits__   = ["Daniel G. A. Smith", "Dominic A. Sirianni"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-05-23"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Set memory & output
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

mol = psi4.geometry(""" 
C    1.39410    0.00000   0.00000
C    0.69705   -1.20732   0.00000
C   -0.69705   -1.20732   0.00000
C   -1.39410    0.00000   0.00000
C   -0.69705    1.20732   0.00000
C    0.69705    1.20732   0.00000
H    2.47618    0.00000   0.00000
H    1.23809   -2.14444   0.00000
H   -1.23809   -2.14444   0.00000
H   -2.47618    0.00000   0.00000
H   -1.23809    2.14444   0.00000
H    1.23809    2.14444   0.00000
symmetry c1
""")

# Basis used in mp2 density fitting
psi4.set_options({'basis': 'aug-cc-pVDZ',
                  'df_basis_scf': 'aug-cc-pvdz-ri'})

check_energy = False

print('\nStarting RHF...')
t = time.time()
RHF_E, wfn = psi4.energy('SCF', return_wfn=True)
print('...RHF finished in %.3f seconds:   %16.10f' % (time.time() - t, RHF_E))

# Grab data from Wavfunction clas
ndocc = wfn.nalpha()
nbf = wfn.nso()
nvirt = nbf - ndocc

# Split eigenvectors and eigenvalues into o and v
eps_occ = np.asarray(wfn.epsilon_a_subset("AO", "ACTIVE_OCC"))
eps_vir = np.asarray(wfn.epsilon_a_subset("AO", "ACTIVE_VIR"))

# Build DF tensors
print('\nBuilding DF ERI tensor Qov...')
t = time.time()
C = wfn.Ca()
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "", "RIFIT", "aug-cc-pvdz")
df = psi4.core.DFTensor(wfn.basisset(), aux, C, ndocc, nvirt) 
# Transformed MO DF tensor
Qov = np.asarray(df.Qov())
print('...Qov build in %.3f seconds with a shape of %s, %.3f GB.' \
% (time.time() - t, str(Qov.shape), np.prod(Qov.shape) * 8.e-9))

print('\nComputing MP2 energy...')
t = time.time()

# At this point we can trivially build the ovov MO tensor and compute MP2
# identically to that as MP2.dat. However, this means we have to build the
# 4-index ERI tensor in memory and results in little gains over conventional
# MP2
# MO = np.einsum('Qia,Qjb->iajb', Qov, Qov)

# A smarter algorithm, loop over occupied indices and exploit ERI symmetry

# This part of the denominator is identical for all i,j pairs
vv_denom = - eps_vir.reshape(-1, 1) - eps_vir

MP2corr_OS = 0.0
MP2corr_SS = 0.0
for i in range(ndocc):
    eps_i = eps_occ[i]
    i_Qv = Qov[:, i, :].copy()
    for j in range(i, ndocc):

        eps_j = eps_occ[j]
        j_Qv = Qov[:, j, :]

        # We can either use einsum here
#        tmp = np.einsum('Qa,Qb->ab', i_Qv, j_Qv)

        # Or a dot product (DGEMM) for speed)
        tmp = np.dot(i_Qv.T, j_Qv)

        # Diagonal elements
        if i == j:
            div = 1.0 / (eps_i + eps_j + vv_denom)
        # Off-diagonal elements
        else:
            div = 2.0 / (eps_i + eps_j + vv_denom)

        # Opposite spin computation
        MP2corr_OS += np.einsum('ab,ab,ab->', tmp, tmp, div)

        # Notice the same-spin compnent has an "exchange" like term associated with it
        MP2corr_SS += np.einsum('ab,ab,ab->', tmp - tmp.T, tmp, div)

print('...finished computing MP2 energy in %.3f seconds.' % (time.time() - t))

MP2corr_E = MP2corr_SS + MP2corr_OS
MP2_E = RHF_E + MP2corr_E

# These are the canonical SCS MP2 coefficients, many others are available however
SCS_MP2corr_E = MP2corr_SS / 3 + MP2corr_OS * (6. / 5)
SCS_MP2_E = RHF_E + SCS_MP2corr_E

print('\nMP2 SS correlation energy:         %16.10f' % MP2corr_SS)
print('MP2 OS correlation energy:         %16.10f' % MP2corr_OS)

print('\nMP2 correlation energy:            %16.10f' % MP2corr_E)
print('MP2 total energy:                  %16.10f' % MP2_E)

print('\nSCS-MP2 correlation energy:        %16.10f' % MP2corr_SS)
print('SCS-MP2 total energy:              %16.10f' % SCS_MP2_E)

if check_energy:
    psi4.energy('MP2')
    psi4.compare_values(psi4.core.get_variable('MP2 TOTAL ENERGY'), MP2_E, 6, 'MP2 Energy')
    psi4.compare_values(psi4.core.get_variable('SCS-MP2 TOTAL ENERGY'), SCS_MP2_E, 6, 'SCS-MP2 Energy')

