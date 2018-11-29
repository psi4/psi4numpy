"""
Density-fitted MP2 from a RHF reference (same as DF-MP2) using a rank-reduced DF tensor
from natural auxiliary functions (NAF) as described in [3].

This is the 'smarter' algorithm described in the paper that avoids the costly
direct contraction of the Coulomb metric with the 3-index integrals (Qov tensor in PSI4 language)
Instead cheap intermediates are used the reduced Qov tensor is regained as the last step.

References: 
1. Algorithm modified from Rob Parrish's most excellent Psi4 plugin example
Bottom of the page: http://www.psicode.org/developers.php
2. Tutorials/03_Hartree-Fock/density-fitting.ipynb
3. M. Kállay, J. Chem. Phys. 2014, 141, 244113. [http://aip.scitation.org/doi/10.1063/1.4905005]
"""

__authors__ = "Holger Kruse"
__credits__ = ["Daniel G. A. Smith", "Dominic A. Sirianni"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-11-29"

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
psi4.set_options({'basis': 'aug-cc-pVDZ', 'df_basis_mp2': 'aug-cc-pvdz-ri'})

check_energy = True

print('\nStarting RHF...')
t = time.time()
RHF_E, wfn = psi4.energy('SCF', return_wfn=True)
print('...RHF finished in %.3f seconds:   %16.10f' % (time.time() - t, RHF_E))

# Grab data from Wavfunction clas
ndocc = wfn.nalpha()
orbital_basis = wfn.basisset()
nbf = wfn.nso()
nvirt = nbf - ndocc

# Split eigenvectors and eigenvalues into o and v
eps_occ = np.asarray(wfn.epsilon_a_subset("AO", "ACTIVE_OCC"))
eps_vir = np.asarray(wfn.epsilon_a_subset("AO", "ACTIVE_VIR"))

# Build DF tensors
print('\nBuilding DF ERI tensor Qov...')
t = time.time()
C = wfn.Ca()

# Build instance of MintsHelper
mints = psi4.core.MintsHelper(orbital_basis)
zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

# build auxiliary basis set object
aux_basis = psi4.core.BasisSet.build(
    mol, "DF_BASIS_MP2", "", "RIFIT",
    psi4.core.get_global_option('df_basis_mp2'))
naux = aux_basis.nbf()

# Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
Ppq = mints.ao_eri(zero_bas, aux_basis, orbital_basis, orbital_basis)

# Build Coulomb metric but only invert, dimension (1, Naux, 1, Naux)
metric = mints.ao_eri(zero_bas, aux_basis, zero_bas, aux_basis)
metric.power(-1.0, 1.e-14)

# Remove excess dimensions of Ppq, & metric
Ppq = np.squeeze(Ppq)
metric = np.squeeze(metric)

# paper uses transpose of Ppq, so we adapt for now
Ppq = np.reshape(Ppq, (naux, nbf * nbf)).T
print("I  = (pq|P) dim:", Ppq.shape)

# cholesky decomp of inverse metric
L = np.linalg.cholesky(metric)
print("L  = cholesky[(P|Q)^1 ]dim:", L.shape)

# Form intermediate W'= I^t*I
Wp = np.dot(Ppq.T, Ppq)
print("W' = (P|P) dim:", Wp.shape)

# form W proper
W = np.dot(np.dot(L.T, Wp), L)
print("W  = (Q|Q) dim:", W.shape)

# from N(bar) from eigenvectors of W
# epsilon threshold is supposed to be in the range of 10^-2 to 10^-4
e_val, e_vec = np.linalg.eigh(W)
eps = 1e-2
print('epsilon = %.3e ' % (eps))
nskipped = 0
Ntmp = np.zeros((naux, naux))
naux2 = 0
for n in range(naux):
    if (abs(e_val[n]) > eps):
        Ntmp[:, naux2] = e_vec[:, n]
        naux2 += 1

print('retaining #naux = %i  of  %i [ %4.1f %% ]' % (naux2, naux,
                                                     naux2 / naux * 100.0))
Nbar = Ntmp[0:naux, 0:naux2]
print("N^bar  = (Q^bar|Q) dim)", Nbar.shape)

# form N'(bar) = L * N(bar)
Npbar = np.dot(L, Nbar)
print("N'^bar  = (P^bar|Q) dim)", Npbar.shape)

# form J(bar) = I * N'(bar)
Jbar = np.dot(Ppq, Npbar)
print("J^bar  = (pq|Q) dim)", Npbar.shape)

# transpose to be inline with PIS4 and expand J(bar) to proper dimensions
# Qpg is then final NAF DF Tensor in AO space
Qpq = Jbar.T.reshape(naux2, nbf, nbf)

# ==> AO->MO transform: Qpq -> Qmo @ O(N^4) <==
Qov = np.einsum('pi,Qpq->Qiq', C, Qpq)
Qov = np.einsum('Qiq,qj->Qij', Qov, C)

# for MP2 we just need the occupied-virtual block of the DF MO tensor
Qov = Qov[:, :ndocc, ndocc:]

time_qov = time.time() - t
print('...Qov build in %.3f seconds with a shape of %s, %.3f GB.' \
% (time_qov, str(Qov.shape), np.prod(Qov.shape) * 8.e-9))

# Having obtained the new MO DF tensor the MP2 energy calculation proceeds as usual
print('\nComputing MP2 energy...')
t = time.time()

# This part of the denominator is identical for all i,j pairs
vv_denom = -eps_vir.reshape(-1, 1) - eps_vir

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

time_mp2 = time.time() - t
print('...finished computing MP2 energy in %.3f seconds.' % (time_mp2))

MP2corr_E = MP2corr_SS + MP2corr_OS
MP2_E = RHF_E + MP2corr_E

# Compute MP2 correlation & total MP2 Energy
print('E(MP2) %f' % (MP2_E))
print('Ecorr(MP2) %f' % (MP2corr_E))

print('NAF-DF-MP2 finished in: %.3f s \n ' % (time_qov + time_mp2))

if check_energy:
    print(' PSI4 MP2 calculation ...')
    # ==> Compare to Psi4 <==
    # re-used RHF wavefunction
    e_total = psi4.energy('mp2', ref_wfn=wfn)
    print('E_REF(MP2) %f' % (e_total))
    ecorr = psi4.core.get_variable('MP2 CORRELATION ENERGY')
    t = time.time()
    print('reference Ecorr(MP2) = %f ; error = %.3e for eps = %.3e' %
          (ecorr, ecorr - MP2corr_E, eps))
    print('PSI4 DF-MP2 finished in %.3f s' \
    % (time.time() - t))
