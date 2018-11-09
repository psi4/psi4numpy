# -*- coding: utf-8 -*-
"""
A simple python script to calculate RHF-CCSD specific rotation in length, 
velocity and modified velocity gauge using coupled cluster linear response theory.

References: 
1. H. Koch and P. Jørgensen, J. Chem. Phys. Volume 93, pp. 3333-3344 (1991).
2. T. B. Pedersen and H. Koch, J. Chem. Phys. Volume 106, pp. 8059-8072 (1997).
3. T. Daniel Crawford, Theor. Chem. Acc., Volume 115, pp. 227-245 (2006).
4. T. B. Pedersen, H. Koch, L. Boman, and A. M. J. Sánchez de Merás, Chem. Phys. Lett.,
   Volime 393, pp. 319, (2004).
5. A Whirlwind Introduction to Coupled Cluster Response Theory, T.D. Crawford, Private Notes,
   (pdf in the current directory).
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
 O     -0.028962160801    -0.694396279686    -0.049338350190                                                                  
 O      0.028962160801     0.694396279686    -0.049338350190                                                                  
 H      0.350498145881    -0.910645626300     0.783035421467                                                                  
 H     -0.350498145881     0.910645626300     0.783035421467                                                                  
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

# convert from nm into hartree
omega = (pc.c * pc.h * 1e9) / (pc.hartree2J * omega_nm)
Om = str(omega)
Om_0 = str(0)

cart = ['X', 'Y', 'Z']
pert = {}
ccpert = {}
tensor = {}
cclinresp = {}
optrot_lg = np.zeros(9)
optrot_vg_om = np.zeros(9)
optrot_vg_0 = np.zeros(9)

###############################################   Length Gauge   ###############################################################

# In length gauge the representation of electric dipole operator is mu i.e. r. So, optical rotation tensor in this gauge
# representation can be given by -Im <<mu;L>>, where L is the angular momemtum operator, refer to Eqn. 5 of [Crawford:2006:227].
# For general form of a response function, refer to Eqn. 94 of [Koch:1991:3333].

print("\n\n Length Gauge Calculations Starting ..\n\n")

# Obtain the required AO Perturabtion Matrices From Mints

# Electric Dipole
dipole_array = ccsd.mints.ao_dipole()

# Angular Momentum
angmom_array = ccsd.mints.ao_angular_momentum()

for i in range(0, 3):
    Mu = "MU_" + cart[i]
    L = "L_" + cart[i]

    # Transform perturbations from AO to MO basis
    pert[Mu] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC,
                         np.asarray(dipole_array[i]))
    pert[L] = -0.5 * np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC,
                               np.asarray(angmom_array[i]))

    # Initializing the perturbation class corresponding to each perturabtion at the given omega
    ccpert[Mu + Om] = HelperCCPert(Mu, pert[Mu], ccsd, cchbar, cclambda, omega)
    ccpert[L + Om] = HelperCCPert(L, pert[L], ccsd, cchbar, cclambda, omega)

    # Solve X and Y amplitudes corresponding to each perturabtion at the given omega
    print(
        '\nsolving right hand perturbed amplitudes for %s @ omega = %s a.u.\n'
        % (Mu, Om))
    ccpert[Mu + Om].solve('right', r_conv=1e-10)

    print(
        '\nsolving left hand perturbed amplitudes for %s @ omega = %s a.u.\n' %
        (Mu, Om))
    ccpert[Mu + Om].solve('left', r_conv=1e-10)

    print(
        '\nsolving right hand perturbed amplitudes for %s @ omega = %s a.u.\n'
        % (L, Om))
    ccpert[L + Om].solve('right', r_conv=1e-10)

    print(
        '\nsolving left hand perturbed amplitudes for %s @ omega = %s a.u.\n' %
        (L, Om))
    ccpert[L + Om].solve('left', r_conv=1e-10)

for A in range(0, 3):
    str_A = "MU_" + cart[A]
    for B in range(0, 3):
        str_B = "L_" + cart[B]
        str_AB = "<<" + str_A + ";" + str_B + ">>"
        str_BA = "<<" + str_B + ";" + str_A + ">>"

        # constructing the linear response functions <<MU;L>> and <<L;MU>> @ given omega
        # The optical rotation tensor beta can be written in length gauge as:
        # beta_pq = 0.5 * (<<MU_p;L_q>>  - <<L_q;MU_p>), Please refer to eq. 49 of 
        # [Pedersen:1997:8059].

        cclinresp[str_AB] = HelperCCLinresp(cclambda, ccpert[str_A + Om],
                                            ccpert[str_B + Om])
        cclinresp[str_BA] = HelperCCLinresp(cclambda, ccpert[str_B + Om],
                                            ccpert[str_A + Om])

        tensor[str_AB] = cclinresp[str_AB].linresp()
        tensor[str_BA] = cclinresp[str_BA].linresp()

        optrot_lg[3 * A + B] = 0.5 * (tensor[str_AB] - tensor[str_BA])

# Isotropic optical rotation in length gauge @ given omega
rlg_au = optrot_lg[0] + optrot_lg[4] + optrot_lg[8]
rlg_au /= 3

print('\n CCSD Optical Rotation Tensor (Length Gauge) @ %d nm' % omega_nm)
print("\t\t%s\t             %s\t                  %s\n" % (cart[0], cart[1],
                                                           cart[2]))

for a in range(0, 3):
    print(" %s %20.10lf %20.10lf %20.10lf\n" %
          (cart[a], optrot_lg[3 * a + 0], optrot_lg[3 * a + 1],
           optrot_lg[3 * a + 2]))

# convert from a.u. into deg/[dm (g/cm^3)]
# refer to eq. 4 of [Crawford:1996:189].
Mass = 0
for atom in range(mol.natom()):
    Mass += mol.mass(atom)
m2a = pc.bohr2angstroms * 1e-10
hbar = pc.h / (2.0 * np.pi)
prefactor = 1e-2 * hbar / (pc.c * 2.0 * np.pi * pc.me * (m2a**2))
prefactor *= prefactor
prefactor *= 288e-30 * (np.pi**2) * pc.na * (pc.bohr2angstroms**4)
prefactor *= -1
specific_rotation_lg = prefactor * rlg_au * omega / Mass
print("Specific rotation @ %d nm (Length Gauge): %10.5lf deg/[dm (g/cm^3)]" %
      (omega_nm, specific_rotation_lg))

###############################################     Velocity Gauge      #########################################################

# In length gauge the representation of electric dipole operator is in terms of p, i,e. the momentum operator.
# So, optical rotation tensor in this gauge representation can be given by -Im <<P;L>>.

print("\n\n Velocity Gauge Calculations Starting ..\n\n")

# Grabbing the momentum integrals from mints
nabla_array = ccsd.mints.ao_nabla()

for i in range(0, 3):
    P = "P_" + cart[i]

    # Transform momentum from AO to MO basis
    pert[P] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC,
                        np.asarray(nabla_array[i]))

    # Initializing the perturbation class
    ccpert[P + Om] = HelperCCPert(P, pert[P], ccsd, cchbar, cclambda, omega)

    # Solve X and Y amplitudes corresponding to the perturabtion at the given omega
    print(
        '\nsolving right hand perturbed amplitudes for %s @ omega = %s a.u.\n'
        % (P, Om))
    ccpert[P + Om].solve('right', r_conv=1e-10)

    print(
        '\nsolving left hand perturbed amplitudes for %s @ omega = %s a.u.\n' %
        (P, Om))
    ccpert[P + Om].solve('left', r_conv=1e-10)

for A in range(0, 3):
    str_A = "P_" + cart[A]
    for B in range(0, 3):
        str_B = "L_" + cart[B]
        str_AB = "<<" + str_A + ";" + str_B + ">>"
        str_BA = "<<" + str_B + ";" + str_A + ">>"

        # constructing the linear response functions <<P;L>> and <<L;P>> @ given omega
        # The optical rotation tensor beta can be written in velocity gauge as:
        # beta_pq = 0.5 * (<<MU_p;L_q>> + <<L_q;MU_p>), Please refer to eq. 49 of 
        # [Pedersen:1991:8059].

        cclinresp[str_AB] = HelperCCLinresp(cclambda, ccpert[str_A + Om],
                                            ccpert[str_B + Om])
        cclinresp[str_BA] = HelperCCLinresp(cclambda, ccpert[str_B + Om],
                                            ccpert[str_A + Om])
        tensor[str_AB] = cclinresp[str_AB].linresp()
        tensor[str_BA] = cclinresp[str_BA].linresp()
        optrot_vg_om[3 * A + B] = 0.5 * (tensor[str_AB] + tensor[str_BA])

# Isotropic optical rotation in velocity gauge @ given omega
rvg_om_au = optrot_vg_om[0] + optrot_vg_om[4] + optrot_vg_om[8]
rvg_om_au /= 3

print('\n CCSD Optical Rotation Tensor (Velocity Gauge) @ %d nm' % omega_nm)
print("\t\t%s\t             %s\t                  %s\n" % (cart[0], cart[1],
                                                           cart[2]))

for a in range(0, 3):
    print(" %s %20.10lf %20.10lf %20.10lf\n" %
          (cart[a], optrot_vg_om[3 * a + 0], optrot_vg_om[3 * a + 1],
           optrot_vg_om[3 * a + 2]))

specific_rotation_vg_om = prefactor * rvg_om_au / Mass
print("Specific rotation @ %d nm (Velocity Gauge): %10.5lf deg/[dm (g/cm^3)]" %
      (omega_nm, specific_rotation_vg_om))

###############################################   Modified Velocity Gauge   ######################################################
#
# Velocity gauge (VG) representation gives a non-zero optical rotation at zero frequency,
# which is clearly an unphysical result. [Pedersen:319:2004] proposed the modified
# velocity gauge (MVG) representation where the VG optical rotation at # zero frequency is subtracted from VG results at a given frequency.

print("\n\nModified Velocity Gauge Calculations Starting ..\n\n")

Om_0 = str(0)
for i in range(0, 3):
    L = "L_" + cart[i]
    P = "P_" + cart[i]
    Om_0 = str(0)

    # Initializing perturbation classes at zero frequency
    ccpert[L + Om_0] = HelperCCPert(L, pert[L], ccsd, cchbar, cclambda, 0)
    ccpert[P + Om_0] = HelperCCPert(P, pert[P], ccsd, cchbar, cclambda, 0)

    # Solving X and Y amplitudes of the perturbation classes at zero frequency

    print(
        '\nsolving right hand perturbed amplitudes for %s @ omega = %s (a.u.)\n'
        % (L, Om_0))
    ccpert[L + Om_0].solve('right', r_conv=1e-10)

    print(
        '\nsolving left hand perturbed amplitudes for %s @ omega = %s (a.u.)\n'
        % (L, Om_0))
    ccpert[L + Om_0].solve('left', r_conv=1e-10)

    print(
        '\nsolving right hand perturbed amplitudes for %s @ omega = %s (a.u.)\n'
        % (P, Om_0))
    ccpert[P + Om_0].solve('right', r_conv=1e-10)

    print(
        '\nsolving left hand perturbed amplitudes for %s @ omega = %s (a.u.)\n'
        % (P, Om_0))
    ccpert[P + Om_0].solve('left', r_conv=1e-10)

for A in range(0, 3):
    str_A = "P_" + cart[A]
    for B in range(0, 3):
        str_B = "L_" + cart[B]
        str_AB = "<<" + str_A + ";" + str_B + ">>"
        str_BA = "<<" + str_B + ";" + str_A + ">>"

        # constructing the linear response functions <<P;L>> and <<L;P>> @ zero frequency)

        cclinresp[str_AB] = HelperCCLinresp(cclambda, ccpert[str_A + Om_0],
                                            ccpert[str_B + Om_0])
        cclinresp[str_BA] = HelperCCLinresp(cclambda, ccpert[str_B + Om_0],
                                            ccpert[str_A + Om_0])

        tensor[str_AB] = cclinresp[str_AB].linresp()
        tensor[str_BA] = cclinresp[str_BA].linresp()

        optrot_vg_0[3 * A + B] = 0.5 * (tensor[str_AB] + tensor[str_BA])

#  MVG(omega) = VG(omega) - VG(0)
optrot_mvg = optrot_vg_om - optrot_vg_0

# Isotropic optical rotation in modified velocity gauge @ given omega
rmvg_au = optrot_mvg[0] + optrot_mvg[4] + optrot_mvg[8]
rmvg_au /= 3

print('\n CCSD Optical Rotation Tensor (Modified Velocity Gauge) @ %d nm' %
      omega_nm)
print("\t\t%s\t             %s\t                  %s\n" % (cart[0], cart[1],
                                                           cart[2]))
for a in range(0, 3):
    print(" %s %20.10lf %20.10lf %20.10lf\n" %
          (cart[a], optrot_vg_0[3 * a + 0], optrot_vg_0[3 * a + 1],
           optrot_vg_0[3 * a + 2]))

specific_rotation_mvg = prefactor * rmvg_au / Mass
print(
    "Specific rotation @ %d nm (Modified Velocity Gauge): %10.5lf deg/[dm (g/cm^3)]"
    % (omega_nm, specific_rotation_mvg))

"""#  Comaprison with PSI4 (if you have near to latest version of psi4)
psi4.set_options({'d_convergence': 1e-10,
                  'e_convergence': 1e-10,
                  'r_convergence': 1e-10,
                  'omega': [589, 'nm'],  
                  'gauge': 'both'})  
psi4.properties('ccsd', properties=['rotation'])
psi4.compare_values(specific_rotation_lg, psi4.get_variable("CCSD SPECIFIC ROTATION (LEN) @ 589NM"), \
 5, "CCSD SPECIFIC ROTATION (LENGTH GAUGE) 589 nm") #TEST
psi4.compare_values(specific_rotation_mvg, psi4.get_variable("CCSD SPECIFIC ROTATION (MVG) @ 589NM"), \
  5, "CCSD SPECIFIC ROTATION (MODIFIED VELOCITY GAUGE) 589 nm") #TEST
"""

psi4.compare_values(specific_rotation_lg, 7.03123, 5,
                    "CCSD SPECIFIC ROTATION (LENGTH GAUGE) 589 nm")  #TEST
psi4.compare_values(
    specific_rotation_mvg, -81.44742, 5,
    "CCSD SPECIFIC ROTATION (MODIFIED VELOCITY GAUGE) 589 nm")  #TEST
