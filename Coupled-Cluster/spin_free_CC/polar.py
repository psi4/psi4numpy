import time
import numpy as np
from helper_cc import helper_ccenergy
from helper_cc import helper_cchbar
from helper_cc import helper_cclambda
from helper_cc import helper_ccpert
from helper_cc import helper_cclinresp

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
ccsd = helper_ccenergy(mol, memory=2)
ccsd.compute_energy()

CCSDcorr_E = ccsd.ccsd_corr_e
CCSD_E = ccsd.ccsd_e

print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
print('Total CCSD energy:                      % 16.15f' % CCSD_E)

cchbar = helper_cchbar(ccsd)

cclambda = helper_cclambda(ccsd,cchbar)
cclambda.compute_lambda()
omega = 0.0

cart = {0:'X', 1: 'Y', 2: 'Z'}
mu = {}
ccpert = {}
polar_AB = {}


for i in range(0,3):
    string = "MU_" + cart[i]
    mu[string] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, np.asarray(ccsd.mints.ao_dipole()[i]))
    ccpert[string] = helper_ccpert(string, mu[string], ccsd, cchbar, cclambda, omega)
    print('\nsolving right hand perturbed amplitudes for %s\n' % string)
    ccpert[string].solve('right')
    print('\nsolving left hand perturbed amplitudes for %s\n'% string)
    ccpert[string].solve('left')
    
print('\n Calculating Polarizability tensor:\n')

for a in range(0,3):
    str_a = "MU_" + cart[a]
    for b in range(0,3):
        str_b = "MU_" + cart[b]
        str_ab = "<<" + str_a + ";" + str_b + ">>"
        polar_AB[str_ab]= helper_cclinresp(cclambda, ccpert[str_a], ccpert[str_b]).linresp()    

print('\nPolarizability tensor (symmetrized):\n')

for a in range(0,3):
    str_a = "MU_" + cart[a]
    for b in range(0,a+1):
        str_b = "MU_" + cart[b]
        str_ab = "<<" + str_a + ";" + str_b + ">>"
        str_ba = "<<" + str_b + ";" + str_a + ">>"
        value = 0.5*(polar_AB[str_ab] + polar_AB[str_ba])
        polar_AB[str_ab] = value
        polar_AB[str_ba] = value
        print(str_ab + ":" + str(value))
    
    
    

