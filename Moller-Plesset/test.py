import time
import numpy as np
np.set_printoptions(precision=12, linewidth=200, suppress=True)
import psi4

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

#mol = psi4.geometry("""
#F     0.0000000000000000  0.0000000000000000 1.76320000000000000
#H     0.0000000000000000  0.0000000000000000 -0.0001000000000000
#symmetry c1
#units bohr
#no_reorient
#no_com
#""")

psi4.core.set_active_molecule(mol)

options = {'BASIS':'STO-3G',
           'SCF_TYPE':'PK',
           'MP2_TYPE':'CONV',
           'E_CONVERGENCE':1e-12,
           'D_CONVERGENCE':1e-12,
           'R_CONVERGENCE':1e-12,
           'print':5,
           'DEBUG':1
           }

psi4.set_options(options)

#scf_grad = psi4.gradient('SCF')
mp2_grad = psi4.gradient('MP2')
modified_ccsd_grad = psi4.gradient('CCSD')
print("Does Modified CCSD = MP2: ", np.allclose(modified_ccsd_grad, mp2_grad), "\n")

