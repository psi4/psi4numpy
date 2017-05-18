import time
import numpy as np
from helper_cc import helper_ccenergy
from helper_cc import helper_cchbar
from helper_cc import helper_cclambda
from helper_cc import helper_ccpert
from helper_cc import helper_cclinresp

np.set_printoptions(precision=15, linewidth=200, suppress=True)
import psi4

#psi4.core.set_memory(int(2e9), False)
psi4.set_memory(int(2e9), False)
psi4.core.set_output_file('output_new.dat', False)

numpy_memory = 2

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

#psi4.set_options({'basis': 'cc-pVDZ'})
psi4.set_options({'basis': 'sto-3g'})

# For numpy
compare_psi4 = True

# Compute CCSD
ccsd = helper_ccenergy(mol, memory=2)
ccsd.compute_energy()

CCSDcorr_E = ccsd.ccsd_corr_e
CCSD_E = ccsd.ccsd_e

print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
print('Total CCSD energy:                      % 16.15f' % CCSD_E)

cchbar = helper_cchbar(ccsd)
#cchbar.build_HBAR()

cclambda = helper_cclambda(ccsd,cchbar)
cclambda.compute_lambda()
omega = 0.0

muX_ao = np.asarray(ccsd.mints.ao_dipole()[0])
muY_ao = np.asarray(ccsd.mints.ao_dipole()[1])
muZ_ao = np.asarray(ccsd.mints.ao_dipole()[2])
muX_mo= np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, muX_ao)
muY_mo= np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, muY_ao)
muZ_mo= np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, muZ_ao)
ccpert_X = helper_ccpert(muX_mo, ccsd, cchbar, cclambda, omega)
ccpert_Y = helper_ccpert(muY_mo, ccsd, cchbar, cclambda, omega)
ccpert_Z = helper_ccpert(muZ_mo, ccsd, cchbar, cclambda, omega)
ccpert_X.solve('right')
ccpert_Y.solve('right')
ccpert_Z.solve('right')

ccpert_X.solve('left')
ccpert_Y.solve('left')
ccpert_Z.solve('left')

cclinresp = helper_cclinresp(cclambda, ccpert_Z, ccpert_Z)
cclinresp.linresp()
print('\nPolarizability --\n')
polar = -1.0 * (cclinresp.polar1 + cclinresp.polar2)
print(polar)
