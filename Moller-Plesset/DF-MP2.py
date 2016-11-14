# A simple Psi 4 input script to compute MP2 from a RHF reference
# Requirements numpy 1.7.2+
#
# Algorithm modified from Rob Parrish's most excellent Psi4 plugin example
# Bottom of the page: http://www.psicode.org/developers.php
#
# Created by: Daniel G. A. Smith
# Date: 2/25/15
# License: GPL v3.0
#


### BETA
import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)

molecule mol {
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
}

set {
basis aug-cc-pVDZ
# Basis used in mp2 density fitting
df_basis_scf aug-cc-pVDZ-ri
}

check_energy = False

print('\nStarting RHF...')
t = time.time()
RHF_E, wfn = energy('SCF', return_wfn=True)
print('...RHF finished in %.3f seconds:   %16.10f' % (time.time() - t, RHF_E))

# Grab data from wavfunction class 
ndocc = wfn.doccpi()[0]

# Split eigenvectors and eigenvalues into o and v
eps_occ = np.asarray(wfn.epsilon_a_subset("AO", "ACTIVE_OCC"))
eps_vir = np.asarray(wfn.epsilon_a_subset("AO", "ACTIVE_VIR"))

print('\nBuilding DF ERI tensor Qov...')
t = time.time()
df = DFTensor(wfn, "DF_BASIS_SCF") 
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
        tmp = np.dot(i_Qv.T, j_Qv)
#        tmp = np.einsum('Qa,Qb->ab', i_Qv, j_Qv)

        # Diagonal elements
        if i == j:
            div = 1.0 / (eps_i + eps_j + vv_denom)
        # Off-diagonal elements
        else:
            div = 2.0 / (eps_i + eps_j + vv_denom)

        MP2corr_OS += np.einsum('ab,ab,ab->', tmp, tmp, div)
        MP2corr_SS += np.einsum('ab,ab,ab->', tmp - tmp.T, tmp, div)

print('...finished computing MP2 energy in %.3f seconds.' % (time.time() - t))

MP2corr_E = MP2corr_SS + MP2corr_OS
MP2_E = RHF_E + MP2corr_E

SCS_MP2corr_E = MP2corr_SS / 3 + MP2corr_OS * (6. / 5)
SCS_MP2_E = RHF_E + SCS_MP2corr_E

print('\nMP2 SS correlation energy:         %16.10f' % MP2corr_SS)
print('MP2 OS correlation energy:         %16.10f' % MP2corr_OS)

print('\nMP2 correlation energy:            %16.10f' % MP2corr_E)
print('MP2 total energy:                  %16.10f' % MP2_E)

print('\nSCS-MP2 correlation energy:        %16.10f' % MP2corr_SS)
print('SCS-MP2 total energy:              %16.10f' % SCS_MP2_E)

if check_energy:
    energy('MP2')
    compare_values(get_variable('MP2 TOTAL ENERGY'), MP2_E, 6, 'MP2 Energy')
    compare_values(get_variable('SCS-MP2 TOTAL ENERGY'), SCS_MP2_E, 6, 'SCS-MP2 Energy')

