"""
Simple demonstration of CQED-RHF method on NaF and NaCl dimers comparing
to reference values provided in [DePrince:2021:094112] and water molecule
from code described in [DePrince:2021:094112]

"""

__authors__   = ["Jon McTague", "Jonathan Foley"]
__credits__   = ["Jon McTague", "Jonathan Foley"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2021-08-19"

# ==> Import Psi4, NumPy, & SciPy <==
import psi4
import numpy as np
from helper_CQED_RHF import *

# Set Psi4 & NumPy Memory Options
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

numpy_memory = 2



# options for H2O
h2o_options_dict = {'basis': 'cc-pVDZ',
               'save_jk': True,
               'scf_type' : 'pk',
               'e_convergence' : 1e-12,
               'd_convergence' : 1e-12
               }


# molecule string for H2O
h2o_string = """

0 1
    O      0.000000000000   0.000000000000  -0.068516219320
    H      0.000000000000  -0.790689573744   0.543701060715
    H      0.000000000000   0.790689573744   0.543701060715
no_reorient
symmetry c1
"""

# energy for H2O from hilbert package described in [DePrince:2021:094112]
expected_h2o_e = -76.016355284146

# electric field for H2O
lam_h2o = np.array([0., 0., 0.05])


# run cqed_rhf on H2O
h2o_dict = cqed_rhf(lam_h2o, h2o_string, h2o_options_dict)
h2o_cqed_rhf_e = h2o_dict['CQED-RHF ENERGY']
h2o_rhf_e = h2o_dict['RHF ENERGY']


print("def2-tzvppd RHF energy of H2O:                    ", h2o_rhf_e)
print("def2-tzvppd CQED-RHF energy of H2o:               ", h2o_cqed_rhf_e)
print("reference def2-tzvppd CQED-RHF energy of H2O:     ", expected_h2o_e)

psi4.compare_values(h2o_cqed_rhf_e, expected_h2o_e,8, 'H2O CQED-RHF E')



