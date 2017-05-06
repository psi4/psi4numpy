# A simple Psi 4 script to compute CCSD from a RHF reference
# Scipy and numpy python modules are required
#
# Algorithms were taken directly from Daniel Crawford's programming website:
# http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming
# Special thanks to Lori Burns for integral help
#
# Created by: Daniel G. A. Smith
# Date: 7/29/14
# License: GPL v3.0
#

import time
import numpy as np
from helper_CCENERGY import *
from helper_CCHBAR import *
from helper_CCLAMBDA import *
from helper_CCPERT import *
from helper_CCLINRESP import *
np.set_printoptions(precision=15, linewidth=200, suppress=True)
import psi4

#psi4.core.set_memory(int(2e9), False)
psi4.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

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
ccsd = helper_CCENERGY(mol, memory=2)
ccsd.compute_energy()

CCSDcorr_E = ccsd.ccsd_corr_e
CCSD_E = ccsd.ccsd_e

print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
print('Total CCSD energy:                      % 16.15f' % CCSD_E)

cchbar = helper_CCHBAR(ccsd)
cchbar.build_HBAR()

cclambda = helper_CCLAMBDA(ccsd,cchbar)
cclambda.compute_lambda()
omega = 0.0

muX_ao = np.asarray(ccsd.mints.ao_dipole()[0])
muY_ao = np.asarray(ccsd.mints.ao_dipole()[1])
muZ_ao = np.asarray(ccsd.mints.ao_dipole()[2])
muX_mo= np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, muX_ao)
muY_mo= np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, muY_ao)
muZ_mo= np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, muZ_ao)
ccpert_X = helper_CCPERT(muX_mo, ccsd, cchbar, cclambda, omega)
ccpert_Y = helper_CCPERT(muY_mo, ccsd, cchbar, cclambda, omega)
ccpert_Z = helper_CCPERT(muZ_mo, ccsd, cchbar, cclambda, omega)
ccpert_X.solve('right')
ccpert_Y.solve('right')
ccpert_Z.solve('right')

ccpert_X.solve('left')
ccpert_Y.solve('left')
ccpert_Z.solve('left')

cclinresp = helper_CCLINRESP(cclambda, ccpert_Z, ccpert_Z)
cclinresp.linresp()
print('\nPolarizability xx\n')
#print('\nFirst term\n')
#print(cclinresp.polar1)
#print('\nSecond term\n')
#print(cclinresp.polar2)
polar = -1.0 * (cclinresp.polar1 + cclinresp.polar2)
print(polar)
