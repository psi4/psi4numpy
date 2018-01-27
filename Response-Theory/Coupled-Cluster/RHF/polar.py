import time
import numpy as np
import sys
sys.path.append("../../../Coupled-Cluster/RHF")
from helper_ccenergy import *
from helper_cchbar   import *    
from helper_cclambda import * 
from helper_ccpert import * 

np.set_printoptions(precision=15, linewidth=200, suppress=True)
import psi4

psi4.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis': 'sto-3g'})
psi4.set_options({'scf_type': 'PK'})
psi4.set_options({'d_convergence': 1e-10})
psi4.set_options({'e_convergence': 1e-10})
psi4.set_options({'r_convergence': 1e-10})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
print('RHF Final Energy                          % 16.10f\n' % rhf_e)

# Calculate Ground State CCSD Equations
ccsd = HelperCCEnergy(mol, rhf_e, rhf_wfn, memory=2)
ccsd.compute_energy(e_conv=1e-10, r_conv=1e-10)
ccsd.compute_energy()

CCSDcorr_E = ccsd.ccsd_corr_e
CCSD_E = ccsd.ccsd_e

print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
print('Total CCSD energy:                      % 16.15f' % CCSD_E)

cchbar = HelperCCHbar(ccsd)

cclambda = HelperCCLambda(ccsd,cchbar)
cclambda.compute_lambda(r_conv=1e-10)

# static polarizability
omega = 0

cart = ['X', 'Y', 'Z']
mu = {}
ccpert = {}
polar_AB = {}


for i in range(0,3):
    string = "MU_" + cart[i]
    mu[string] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, np.asarray(ccsd.mints.ao_dipole()[i]))
    ccpert[string] = HelperCCPert(string, mu[string], ccsd, cchbar, cclambda, omega)
    print('\nsolving right hand perturbed amplitudes for %s\n' % string)
    ccpert[string].solve('right', r_conv=1e-10)
    print('\nsolving left hand perturbed amplitudes for %s\n'% string)
    ccpert[string].solve('left', r_conv=1e-10)

print("\nComputing <<Mu;Mu>_(%1.5lf) tensor." % omega)

for a in range(0,3):
    str_a = "MU_" + cart[a]
    for b in range(0,3):
        str_b = "MU_" + cart[b]
        polar_AB[3*a+b]  = HelperCCLinresp(cclambda, ccpert[str_a], ccpert[str_b]).linresp()

# Symmetrizing the tensor
for a in range(0,3):
    for b in range(0,a+1):
        ab = 3*a+b
        ba = 3*b+a
        if a != b:    
            polar_AB[ab] = 0.5*(polar_AB[ab] + polar_AB[ba])
            polar_AB[ba] = polar_AB[ab]    

print('\n CCSD Dipole Polarizability Tensor (Symmetrized) at omega = %8.6lf a.u \n'% omega)
print("\t\t%s\t             %s\t                  %s\n" % (cart[0], cart[1], cart[2]))    
for a in range(0,3):
    print(" %s %20.10lf %20.10lf %20.10lf\n" % ( cart[a], polar_AB[3*a+0], polar_AB[3*a+1], polar_AB[3*a+2]))    

trace = polar_AB[0] + polar_AB[4] + polar_AB[8]
Isotropic_polar = trace/3.0
        
# PSI4's polarizability
psi4.set_options({'omega': [0, 'ev']})
psi4.properties('ccsd', properties=['polarizability'])
psi4.compare_values(Isotropic_polar, psi4.get_variable("CCSD DIPOLE POLARIZABILITY @ INF NM"),  3, "CCSD Isotropic Dipole Polarizability @ Inf nm") #TEST
