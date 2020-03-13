"""
A reference implementation of MP2 using the Laplace transformation for a restricted reference.
This is performed in an MO basis for demonstration, but LT-MP2 can be extended to AO-LT-MP2.

References:
    J. Almlof, Chem. Phys. Lett. 181, 319 (1991).
    P. Y. Ayala and G. E. Scuseria, J. Chem. Phys., 110, 3660 (1999).
"""

__authors__ = "Oliver J. Backhouse"
__credits__ = ["Oliver J. Backhouse"]

__copyright__ = "(c) 2014-2020, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-03-01"

import time
import numpy as np
import psi4

# Settings
compare_to_psi4 = True
grid_size = 40

# Set the memory and output file
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Set molecule and basis
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis': 'aug-cc-pvdz'})

# Perform SCF
print('\nPerforming SCF...')
e_scf, wfn = psi4.energy('SCF', return_wfn=True)
mints = psi4.core.MintsHelper(wfn.basisset())

# Get occupied and virtual MO energies and coefficients
e_occ = wfn.epsilon_a_subset('AO', 'ACTIVE_OCC').np 
e_vir = wfn.epsilon_a_subset('AO', 'ACTIVE_VIR').np
c_occ = wfn.Ca_subset('AO', 'OCC')
c_vir = wfn.Ca_subset('AO', 'VIR')

# Get the two-electron integrals in MO basis
print('Building MO integrals...')
iajb = mints.mo_eri(c_occ, c_vir, c_occ, c_vir).np

# Build the grids and weights according to Gauss-Laguerre quadrature
# Also apply a weighting function w(x) = exp(x)
grid, weights = np.polynomial.laguerre.laggauss(grid_size)
weights *= np.exp(grid)

# Loop over grid points and compute energies
e_mp2_corr_os = 0.0
e_mp2_corr_ss = 0.0

print('Looping over %d grid points...' % grid_size)
t_start = time.time()
for t, w in zip(grid, weights):
    # Build the amplitudes, including the contribution from the Laplace-transformed term
    # In some equations, this is combined with the next einsum for the energy contributions,
    # instead we contract the amplitudes with the MO integrals to get the energies.
    t_occ = np.exp( t * e_occ)
    t_vir = np.exp(-t * e_vir)
    iajb_t = np.einsum('i,a,j,b,iajb->iajb', t_occ, t_vir, t_occ, t_vir, iajb)

    # Calculate MP2 energy for spin cases
    e_mp2_corr_os_contr = np.einsum('iajb,iajb->', iajb_t, iajb)
    e_mp2_corr_ss_contr = np.einsum('iajb,iajb->', iajb_t, iajb - iajb.swapaxes(1, 3))

    # Add to total including weights
    e_mp2_corr_os -= w * e_mp2_corr_os_contr
    e_mp2_corr_ss -= w * e_mp2_corr_ss_contr

e_mp2_corr = e_mp2_corr_os + e_mp2_corr_ss
e_mp2 = e_scf + e_mp2_corr

e_scs_mp2_corr = e_mp2_corr_os * (6. / 5) + e_mp2_corr_ss / 3
e_scs_mp2 = e_scf + e_scs_mp2_corr

print('MP2 energy calculated in %.3f seconds.\n' % (time.time() - t_start))

print('\nMP2 SS correlation energy:  %16.10f' % e_mp2_corr_ss)
print('MP2 OS correlation energy:  %16.10f' % e_mp2_corr_os)
                                   
print('\nMP2  correlation energy:    %16.10f' % e_mp2_corr)
print('MP2 total energy:           %16.10f' % e_mp2)

print('\nSCS-MP2 correlation energy: %16.10f' % e_scs_mp2_corr)
print('SCS-MP2 total energy:       %16.10f\n' % e_scs_mp2)

if compare_to_psi4:
    psi4.energy('MP2')
    psi4.compare_values(psi4.core.variable('MP2 TOTAL ENERGY'), e_mp2, 4, 'MP2 Energy')
    psi4.compare_values(psi4.core.variable('SCS-MP2 TOTAL ENERGY'), e_scs_mp2, 4, 'SCS-MP2 Energy')

