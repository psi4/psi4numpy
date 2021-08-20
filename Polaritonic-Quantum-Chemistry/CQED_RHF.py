"""
Simple demonstration of CQED-RHF method on the water molecule
coupled to a strong photon field with comparison to results from 
code in the hilbert package described in [DePrince:2021:094112] and available
at https://github.com/edeprince3/hilbert

"""

__authors__ = ["Jon McTague", "Jonathan Foley"]
__credits__ = ["Jon McTague", "Jonathan Foley"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2021-08-19"

# ==> Import Psi4, NumPy, and helper_CQED_RHF <==
import psi4
import numpy as np
from helper_CQED_RHF import *

# Set Psi4 & NumPy Memory Options
psi4.set_memory("2 GB")
psi4.core.set_output_file("output.dat", False)

numpy_memory = 2


# options for H2O
h2o_options_dict = {
    "basis": "cc-pVDZ",
    "save_jk": True,
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
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

# electric field for H2O - polarized along z-axis with mangitude 0.05 atomic units
lam_h2o = np.array([0.0, 0.0, 0.05])


# run cqed_rhf on H2O
h2o_dict = cqed_rhf(lam_h2o, h2o_string, h2o_options_dict)

# parse dictionary for ordinary RHF and CQED-RHF energy
h2o_cqed_rhf_e = h2o_dict["CQED-RHF ENERGY"]
h2o_rhf_e = h2o_dict["RHF ENERGY"]


print("\n    RHF Energy:                %.8f" % h2o_rhf_e)
print("    CQED-RHF Energy:           %.8f" % h2o_cqed_rhf_e)
print("    Reference CQED-RHF Energy: %.8f\n" % expected_h2o_e)

psi4.compare_values(h2o_cqed_rhf_e, expected_h2o_e, 8, "H2O CQED-RHF E")
