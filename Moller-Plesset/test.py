import time
import numpy as np
np.set_printoptions(precision=8, linewidth=200, suppress=True)
import psi4

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.core.set_active_molecule(mol)

options = {'BASIS':'STO-3G', #'SCF_TYPE':'DF',
           'E_CONVERGENCE':1e-10,
           'D_CONVERGENCE':1e-10,
           'print':9}

psi4.set_options(options)

mp2_grad = psi4.gradient('MP2')
#print("MP2 Correction = ",mp2_e - rhf_e)

mp2_grad.print_out()
