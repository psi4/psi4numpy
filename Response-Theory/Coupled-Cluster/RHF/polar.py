# -*- coding: utf-8 -*-
"""
A simple python script to calculate RHF-CCSD electric dipole polarizabilities in length
gauge using coupled cluster linear response theory.

References: 
- Equations and algorithms from [Koch:1991:3333] and [Crawford:xxxx]
"""

__authors__ = "Ashutosh Kumar"
__credits__ = [
    "Ashutosh Kumar", "Daniel G. A. Smith", "Lori A. Burns", "T. D. Crawford"
]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-02-20"

import os.path
import sys
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '../../../Coupled-Cluster/RHF'))
import numpy as np
np.set_printoptions(precision=15, linewidth=200, suppress=True)
# Import all the coupled cluster utilities
from helper_ccenergy import *
from helper_cchbar import *
from helper_cclambda import *
from helper_ccpert import *

import psi4
from psi4 import constants as pc

psi4.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

# can only handle C1 symmetry
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1  
""")

# setting up SCF options
psi4.set_options({
    'basis': 'sto-3g',
    'scf_type': 'PK',
    'd_convergence': 1e-10,
    'e_convergence': 1e-10,
})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

print('RHF Final Energy                          % 16.10f\n' % rhf_e)

# Calculate Ground State CCSD energy
ccsd = HelperCCEnergy(mol, rhf_e, rhf_wfn, memory=2)
ccsd.compute_energy(e_conv=1e-10, r_conv=1e-10)

CCSDcorr_E = ccsd.ccsd_corr_e
CCSD_E = ccsd.ccsd_e

print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
print('Total CCSD energy:                      % 16.15f' % CCSD_E)

# Now that we have T1 and T2 amplitudes, we can construct
# the pieces of the similarity transformed hamiltonian (Hbar).
cchbar = HelperCCHbar(ccsd)

# Calculate Lambda amplitudes using Hbar
cclambda = HelperCCLambda(ccsd, cchbar)
cclambda.compute_lambda(r_conv=1e-10)

# frequency of calculation
omega_nm = 589
# conver from nm into hartree
omega_nm = 589
omega = (pc.c * pc.h * 1e9) / (pc.hartree2J * omega_nm)

cart = ['X', 'Y', 'Z']
Mu = {}
ccpert = {}
polar_AB = {}

# Obtain AO Dipole Matrices From Mints
dipole_array = ccsd.mints.ao_dipole()

for i in range(0, 3):
    string = "MU_" + cart[i]

    # Transform dipole integrals from AO to MO basis
    Mu[string] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC,
                           np.asarray(dipole_array[i]))

    # Initializing the perturbation classs corresponding to dipole perturabtion at the given omega
    ccpert[string] = HelperCCPert(string, Mu[string], ccsd, cchbar, cclambda,
                                  omega)

    # Solve X and Y amplitudes corresponding to dipole perturabtion at the given omega
    print('\nsolving right hand perturbed amplitudes for %s\n' % string)
    ccpert[string].solve('right', r_conv=1e-10)

    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
    ccpert[string].solve('left', r_conv=1e-10)

# Please refer to eq. 94 of [Koch:1991:3333] for the general form of linear response functions.
# For electric dipole polarizabilities, A = mu[x/y/z] and B = mu[x/y/z],
# Ex. alpha_xy = <<mu_x;mu_y>>, where mu_x = x and mu_y = y

print("\nComputing <<Mu;Mu> tensor @ %d nm" % omega_nm)

for a in range(0, 3):
    str_a = "MU_" + cart[a]
    for b in range(0, 3):
        str_b = "MU_" + cart[b]
        # Computing the alpha tensor
        polar_AB[3 * a + b] = HelperCCLinresp(cclambda, ccpert[str_a],
                                              ccpert[str_b]).linresp()

# Symmetrizing the tensor
for a in range(0, 3):
    for b in range(0, a + 1):
        ab = 3 * a + b
        ba = 3 * b + a
        if a != b:
            polar_AB[ab] = 0.5 * (polar_AB[ab] + polar_AB[ba])
            polar_AB[ba] = polar_AB[ab]

print(
    '\nCCSD Dipole Polarizability Tensor (Length Gauge) at omega = %8.6lf au, %d nm\n'
    % (omega, omega_nm))
print("\t\t%s\t             %s\t                  %s\n" % (cart[0], cart[1],
                                                           cart[2]))

for a in range(0, 3):
    print(" %s %20.10lf %20.10lf %20.10lf\n" %
          (cart[a], polar_AB[3 * a + 0], polar_AB[3 * a + 1],
           polar_AB[3 * a + 2]))

# Isotropic polarizability
trace = polar_AB[0] + polar_AB[4] + polar_AB[8]
Isotropic_polar = trace / 3.0

print(
    " Isotropic CCSD Dipole Polarizability @ %d nm (Length Gauge): %20.10lf a.u."
    % (omega_nm, Isotropic_polar))

"""# Comaprison with PSI4 (if you have near to latest version of psi4)
psi4.set_options({'d_convergence': 1e-10})
psi4.set_options({'e_convergence': 1e-10})
psi4.set_options({'r_convergence': 1e-10})
psi4.set_options({'omega': [589, 'nm']})
psi4.properties('ccsd', properties=['polarizability'])
psi4.compare_values(Isotropic_polar, psi4.get_variable("CCSD DIPOLE POLARIZABILITY @ 589NM"),  6, "CCSD Isotropic Dipole Polarizability @ 589 nm (Length Gauge)") #TEST
"""

psi4.compare_values(
    Isotropic_polar, 3.359649777784, 6,
    "CCSD Isotropic Dipole Polarizability @ 589 nm (Length Gauge)")  #TEST
