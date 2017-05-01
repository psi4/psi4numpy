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
from helper_CC_spin_free import *
from helper_CC import *
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

psi4.core.set_memory(int(2e9), False)
psi4.core.set_output_file('ccsd_sf.dat', False)

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
ccsd = helper_CCSD_SF(mol, memory=2)
ccsd.compute_energy()

CCSDcorr_E = ccsd.ccsd_corr_e
CCSD_E = ccsd.ccsd_e

print('\nFinal CCSD correlation energy:          % 16.10f' % CCSDcorr_E)
print('Total CCSD energy:                      % 16.10f' % CCSD_E)

ccsd = helper_CCSD(mol, memory=2)
ccsd.compute_energy()

CCSDcorr_E = ccsd.ccsd_corr_e
CCSD_E = ccsd.ccsd_e

print('\nFinal CCSD correlation energy:          % 16.10f' % CCSDcorr_E)
print('Total CCSD energy:                      % 16.10f' % CCSD_E)
