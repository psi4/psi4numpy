# ==> Import Psi4, NumPy, & SciPy <==
import psi4
import numpy as np
from helper_cqed_rhf import *

# Set Psi4 & NumPy Memory Options
psi4.set_memory('2 GB')
#psi4.core.set_output_file('output.dat', False)

numpy_memory = 2

psi4_options_dict = {
                  'basis':        'def2-tzvppd',
                  'scf_type':     'pk',
                  'reference':    'rhf',
                  'mp2_type':     'conv',
                  'save_jk': True,
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10
                  }
# basis set etc
psi4.set_options(psi4_options_dict)

NaF_string = """

0 1
    NA           0.000000000000     0.000000000000    -0.875819904077
    F            0.000000000000     0.000000000000     1.059820520433
no_reorient
#nocom
symmetry c1
"""

NaCl_string = """

0 1
    NA           0.000000000000     0.000000000000    -1.429419641344
    CL           0.000000000000     0.000000000000     0.939751385626
no_reorient
#nocom
symmetry c1
"""

expected_NaF =  -261.371070718358
expected_NaCl = -621.438985539266

# electric field
Ex = 0.
Ey = 0.
Ez = 0.01

lam = np.array([Ex, Ey, Ez])

# run cqed_rhf on NaF and compare to expected answer
cqed_rhf_dict = cqed_rhf(lam, NaF_string, psi4_options_dict)
em_cqed_rhf_e = cqed_rhf_dict['CQED-RHF ENERGY']
em_rhf_e = cqed_rhf_dict['RHF ENERGY']
assert np.isclose(em_cqed_rhf_e, expected_NaF,5e-5)
print("def2-tzvppd RHF energy of NaF:               ", em_rhf_e)
print("def2-tzvppd CQED-RHF energy of NaF:          ", em_cqed_rhf_e)
print("reference def2-tzvppd CQED-RHF energy of NaF:", expected_NaF)

cqed_rhf_dict = cqed_rhf(lam, NaCl_string, psi4_options_dict)
em_rhf_e = cqed_rhf_dict['RHF ENERGY']
em_cqed_rhf_e = cqed_rhf_dict['CQED-RHF ENERGY']
assert np.isclose(em_cqed_rhf_e, expected_NaCl,5e-5)
print("def2-tzvppd RHF energy of NaCl:               ", em_rhf_e)
print("def2-tzvppd CQED-RHF energy of NaCl:          ", em_cqed_rhf_e)
print("reference def2-tzvppd CQED-RHF energy of NaCl:", expected_NaCl)
