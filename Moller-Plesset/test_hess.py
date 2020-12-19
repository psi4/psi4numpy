import time
import numpy as np
import psi4
import copy

# Setup NumPy options
np.set_printoptions(
    precision=14, 
    linewidth=200, 
    suppress=True, 
    threshold=10000
)

# Specify Molecule
mol = psi4.geometry("""
O
H 1 R
H 1 R 2 104
symmetry c1
""")

# physical constants changed, so geometry changes slightly
from pkg_resources import parse_version
if parse_version(psi4.__version__) >= parse_version("1.3a1"):
    mol.R = 1.1 * 0.52917721067 / 0.52917720859
else:
    mol.R = 1.1

psi4.core.set_active_molecule(mol)

# Set Psi4 Options
options = {
    'BASIS': 'STO-3G',
    'SCF_TYPE': 'PK',
    'MP2_TYPE': 'CONV',
    'E_CONVERGENCE': 1e-14,
    'D_CONVERGENCE': 1e-14,
    'print': 100
}

psi4.set_options(options)

scf_hess = psi4.hessian('SCF')
