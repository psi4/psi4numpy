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
#psi4.core.set_output_file('output.dat', False)

numpy_memory = 2

# options for NaF and NaCl
nax_options_dict = {
                  'basis':        'def2-tzvppd',
                  'scf_type':     'pk',
                  'reference':    'rhf',
                  'mp2_type':     'conv',
                  'save_jk': True,
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10
                  }

# options for H2O
h2o_options_dict = {'basis': 'cc-pVDZ',
               'save_jk': True,
               'scf_type' : 'pk',
               'e_convergence' : 1e-12,
               'd_convergence' : 1e-12
               }

# molecule string for NaF
naf_string = """

0 1
    NA           0.000000000000     0.000000000000    -0.875819904077
    F            0.000000000000     0.000000000000     1.059820520433
no_reorient
#nocom
symmetry c1
"""
# molecule string for NaCl
nacl_string = """

0 1
    NA           0.000000000000     0.000000000000    -1.429419641344
    CL           0.000000000000     0.000000000000     0.939751385626
no_reorient
#nocom
symmetry c1
"""

# molecule string for H2O
h2o_string = """

0 1
    O      0.000000000000   0.000000000000  -0.068516219320
    H      0.000000000000  -0.790689573744   0.543701060715
    H      0.000000000000   0.790689573744   0.543701060715
no_reorient
symmetry c1
"""

# raw energies for NaF and NaCl from [DePrince:2021:094112] which 
# used density fittings, so they should match to 10 mH or so
expected_naf_e =  -261.371070718358
expected_nacl_e = -621.438985539266

# energy for H2O from hilbert package described in [DePrince:2021:094112]
expected_h2o_e = -76.016355284146

# electric field for NaF and NaCl
lam_nax = np.array([0., 0., 0.01])

# electric field for H2O
lam_h2o = np.array([0., 0., 0.05])

# run cqed_rhf on NaF 
naf_dict = cqed_rhf(lam_nax, naf_string, nax_options_dict)
naf_cqed_rhf_e = naf_dict['CQED-RHF ENERGY']
naf_rhf_e = naf_dict['RHF ENERGY']

# run cqed_rhf on NaF 
nacl_dict = cqed_rhf(lam_nax, nacl_string, nax_options_dict)
nacl_cqed_rhf_e = nacl_dict['CQED-RHF ENERGY']
nacl_rhf_e = nacl_dict['RHF ENERGY']

# run cqed_rhf on H2O
h2o_dict = cqed_rhf(lam_h2o, h2o_string, h2o_options_dict)
h2o_cqed_rhf_e = h2o_dict['CQED-RHF ENERGY']
h2o_rhf_e = h2o_dict['RHF ENERGY']

#assert np.isclose(em_cqed_rhf_e, expected_NaF,5e-5)
print("def2-tzvppd RHF energy of NaF:                    ", naf_rhf_e)
print("def2-tzvppd CQED-RHF energy of NaF:               ", naf_cqed_rhf_e)
print("reference def2-tzvppd CQED-RHF energy of NaF (DF):", expected_naf_e)

print("def2-tzvppd RHF energy of NaCl:                    ", nacl_rhf_e)
print("def2-tzvppd CQED-RHF energy of NaCl:               ", nacl_cqed_rhf_e)
print("reference def2-tzvppd CQED-RHF energy of NaCl (DF):", expected_nacl_e)

print("def2-tzvppd RHF energy of H2O:                    ", h2o_rhf_e)
print("def2-tzvppd CQED-RHF energy of H2o:               ", h2o_cqed_rhf_e)
print("reference def2-tzvppd CQED-RHF energy of H2O:     ", expected_h2o_e)

psi4.compare_values(naf_cqed_rhf_e, expected_naf_e,4, 'NaF CQED-RHF E')
psi4.compare_values(nacl_cqed_rhf_e, expected_nacl_e,4, 'NaCl CQED-RHF E')
psi4.compare_values(h2o_cqed_rhf_e, expected_h2o_e,8, 'H2O CQED-RHF E')



