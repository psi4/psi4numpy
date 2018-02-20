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
from psi4 import constants as pc

psi4.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

mol = psi4.geometry("""
 O     -0.028962160801    -0.694396279686    -0.049338350190                                                                  
 O      0.028962160801     0.694396279686    -0.049338350190                                                                  
 H      0.350498145881    -0.910645626300     0.783035421467                                                                  
 H     -0.350498145881     0.910645626300     0.783035421467                                                                  
symmetry c1        
""")

psi4.set_options({'basis': 'sto-3g'})
psi4.set_options({'scf_type':'PK'})
psi4.set_options({'e_convergence':1e-10})
psi4.set_options({'d_convergence':1e-10})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)  
print('RHF Final Energy                          % 16.10f\n' % rhf_e)


# Calculate Ground State CCSD Equations
ccsd = HelperCCEnergy(mol, rhf_e, rhf_wfn, memory=2)
ccsd.compute_energy(e_conv=1e-10, r_conv=1e-10)

CCSDcorr_E = ccsd.ccsd_corr_e
CCSD_E = ccsd.ccsd_e

print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
print('Total CCSD energy:                      % 16.15f' % CCSD_E)

cchbar = HelperCCHbar(ccsd)

cclambda = HelperCCLambda(ccsd,cchbar)
cclambda.compute_lambda(r_conv=1e-10)

# nm into hartree
omega_nm = 589 
omega = (pc.c * pc.h * 1e9)/(pc.hartree2J * omega_nm)

cart = ['X', 'Y', 'Z']
pert = {}
ccpert = {}
tensor= {}
cclinresp = {}
optrot_tensor={}


dipole_array = ccsd.mints.ao_dipole()
angmom_array = ccsd.mints.ao_angular_momentum()

for i in range(0,3):
    string_Mu = "MU_" + cart[i]
    string_L = "L_" + cart[i]
    pert[string_Mu] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, np.asarray(dipole_array[i]))
    pert[string_L] = -0.5 * np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, np.asarray(angmom_array[i]))
    ccpert[string_Mu] = HelperCCPert(string_Mu, pert[string_Mu], ccsd, cchbar, cclambda, omega)
    ccpert[string_L] = HelperCCPert(string_L, pert[string_L], ccsd, cchbar, cclambda, omega)
    print('\nsolving right hand perturbed amplitudes for %s\n' % string_Mu)
    ccpert[string_Mu].solve('right', r_conv=1e-10)
    print('\nsolving left hand perturbed amplitudes for %s\n'% string_Mu)
    ccpert[string_Mu].solve('left', r_conv=1e-10)
    print('\nsolving right hand perturbed amplitudes for %s\n' % string_L)
    ccpert[string_L].solve('right', r_conv=1e-10)
    print('\nsolving left hand perturbed amplitudes for %s\n'% string_L)
    ccpert[string_L].solve('left', r_conv=1e-10)

print("\nComputing optical rotation tensor (LENGTH GAUGE) @ %d nm" % omega_nm)

for a in range(0,3):
    str_a = "MU_" + cart[a]
    for b in range(0,3):
        str_b = "L_" + cart[b]
        str_ab = "<<" + str_a + ";" + str_b + ">>"
        str_ba = "<<" + str_b + ";" + str_a + ">>"
        cclinresp[str_ab]= HelperCCLinresp(cclambda, ccpert[str_a], ccpert[str_b])
        cclinresp[str_ba]= HelperCCLinresp(cclambda, ccpert[str_b], ccpert[str_a])
        tensor[str_ab]= cclinresp[str_ab].linresp()
        tensor[str_ba]= cclinresp[str_ba].linresp()
        optrot_tensor[3*a+b] = 0.5 * (tensor[str_ab] - tensor[str_ba])

print('\n CCSD Optical Rotation Tensor (Length gauge) @ %d nm'%omega_nm)
print("\t\t%s\t             %s\t                  %s\n" % (cart[0], cart[1], cart[2]))
for a in range(0,3):
    print(" %s %20.10lf %20.10lf %20.10lf\n" % ( cart[a], optrot_tensor[3*a+0], optrot_tensor[3*a+1], optrot_tensor[3*a+2]))

rotation_au = optrot_tensor[0] + optrot_tensor[4] + optrot_tensor[8]
rotation_au /= 3 

# convert into specific rotation units
Mass = 0
for atom in range(mol.natom()):
    Mass += mol.mass(atom)
m2a = pc.bohr2angstroms * 1e-10
hbar = pc.h/(2.0 * np.pi)
prefactor  = 1e-2 * hbar/(pc.c * 2.0 * np.pi * pc.me * (m2a **2))
prefactor *= prefactor   
prefactor *= 288e-30 * (np.pi **2) * pc.na * (pc.bohr2angstroms **4)
prefactor *= -1
specific_rotation = prefactor * rotation_au * omega/Mass
print("Specific rotation (deg/[dm (g/cm^3)]) @ %d nm (LENGTH GAUGE): %10.5lf"% (omega_nm, specific_rotation))

#  Comaprison with PSI4
psi4.set_options({'d_convergence': 1e-10})
psi4.set_options({'e_convergence': 1e-10})
psi4.set_options({'r_convergence': 1e-10})
psi4.set_options({'omega': [589, 'nm']})
psi4.set_options({'gauge': 'length'})
psi4.properties('ccsd', properties=['rotation'])
psi4.compare_values(specific_rotation, psi4.get_variable("CCSD SPECIFIC ROTATION (LEN) @ 589NM"),  5, "CCSD SPECIFIC ROTATION (LENGTH GAUGE) 589 nm") #TEST
