#import time
#import numpy as np
#import psi4
#import copy
#
## Setup NumPy options
#np.set_printoptions(
#    precision=12, 
#    linewidth=200, 
#    suppress=True, 
#    threshold=np.nan
#)
#
## Specify Molecule
#mol = psi4.geometry("""
#O
#H 1 R
#H 1 R 2 104
#symmetry c1
#""")
#
## physical constants changed, so geometry changes slightly
#from pkg_resources import parse_version
#if parse_version(psi4.__version__) >= parse_version("1.3a1"):
#    mol.R = 1.1 * 0.52917721067 / 0.52917720859
#else:
#    mol.R = 1.1
#
#psi4.core.set_active_molecule(mol)
#
## Set Psi4 Options
#options = {
#    'BASIS': 'STO-3G',
#    'SCF_TYPE': 'PK',
#    'MP2_TYPE': 'CONV',
#    'E_CONVERGENCE': 1e-12,
#    'D_CONVERGENCE': 1e-12,
#    'print': 10
#}
#
#psi4.set_options(options)
#
## Perform MP2 Energy Calculation
##mp2_e, wfn = psi4.energy('MP2', return_wfn=True)
##mp2_grad = psi4.gradient('SCF')
#mp2_hess = psi4.hessian('MP2')
#
#
#
#
# -*- coding: utf-8 -*-
"""
This script calculates nuclear hessians for MP2 using the
gradients of one and two electron integrals obtained from PSI4. 

References: 
1. "Derivative studies in hartree-fock and møller-plesset theories",
J. A. Pople, R. Krishnan, H. B. Schlegel and J. S. Binkley
DOI: 10.1002/qua.560160825

2. "Analytic evaluation of second derivatives using second-order many-body 
perturbation theory and unrestricted Hartree-Fock reference functions",
J. F. Stanton, J. Gauss, and R. J. Bartlett
DOI: 10.1016/0009-2614(92)86135-5

3. "Coupled-cluster open shell analytic gradients: Implementation of the
direct product decomposition approach in energy gradient calculations",
J. Gauss, J. F. Stanton, R. J. Bartlett
DOI: 10.1063/1.460915
"""

__authors__ = "Kirk C. Pearce"
__credits__ = ["Kirk C. Pearce", "Ashutosh Kumar"]
__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2019-02-11"

import time
import numpy as np
#import psi4
import copy

import psi4
from psi4 import *
from psi4.core import *
from psi4.driver.diatomic import anharmonicity
from psi4.driver.gaussian_n import *
from psi4.driver.frac import ip_fitting, frac_traverse
from psi4.driver.aliases import *
from psi4.driver.driver_cbs import *
from psi4.driver.wrapper_database import database, db, DB_RGT, DB_RXN
from psi4.driver.wrapper_autofrag import auto_fragments
from psi4.driver.constants.physconst import *
psi4_io = core.IOManager.shared_object()


# Setup NumPy options
np.set_printoptions(
    precision=15, 
    linewidth=200, 
    suppress=True, 
    threshold=np.nan
)

psi4.set_memory(int(1e9), False)
#psi4.core.set_output_file('output.dat', False)
psi4.core.set_num_threads(4)

# Specify Molecule
#mol = psi4.geometry("""
#H
#F 1 R
#units bohr
#symmetry c1
#""")
mol = psi4.geometry("""
O
H 1 R
H 1 R 2 104
units bohr
symmetry c1
""")
#mol = psi4.geometry("""
#O   0.00000000     0.00000000     0.07579184
#H   0.00000000    -0.86681183    -0.60143578
#H   0.00000000     0.86681183    -0.60143578
#symmetry c1
#units bohr
#noreorient
#""")

# physical constants changed, so geometry changes slightly
from pkg_resources import parse_version
mol.R = 1.1
#if parse_version(psi4.__version__) >= parse_version("1.3a1"):
#    mol.R = 1.1 * 0.52917721067 / 0.52917720859
#else:
#    mol.R = 1.1

psi4.core.set_active_molecule(mol)

# Set Psi4 Options
options = {
    #'BASIS': 'STO-3G',
    'SCF_TYPE': 'PK',
    'GUESS': 'CORE',
    'MP2_TYPE': 'CONV',
    'E_CONVERGENCE': 1e-14,
    'D_CONVERGENCE': 1e-14,
    'print': 1
}

# Define custom basis set
def basisspec_psi4_yo__anonymous203a4092(mol, role):
    basstrings = {}
    mol.set_basis_all_atoms("sto-3g", role=role)
    basstrings['sto-3g'] = """
cartesian
****
H     0
S   3   1.00
3.4252509             0.1543290
0.6239137             0.5353281
0.1688554             0.4446345
****
O     0
S   3   1.00
130.7093214              0.1543290
23.8088661              0.5353281
6.4436083              0.4446345
SP   3   1.00
5.0331513             -0.0999672             0.1559163
1.1695961              0.3995128             0.6076837
0.3803890              0.7001155             0.3919574
****
F     0
S   3   1.00
166.6791340              0.1543290
30.3608123              0.5353281
8.2168207              0.4446345
SP   3   1.00
6.4648032             -0.0999672             0.1559163
1.5022812              0.3995128             0.6076837
0.4885885              0.7001155             0.3919574
****
"""
    return basstrings
qcdb.libmintsbasisset.basishorde['ANONYMOUS203A4092'] = basisspec_psi4_yo__anonymous203a4092
core.set_global_option("BASIS", "anonymous203a4092")

psi4.set_options(options)

# Perform MP2 Energy Calculation
mp2_hess = psi4.hessian('MP2')
