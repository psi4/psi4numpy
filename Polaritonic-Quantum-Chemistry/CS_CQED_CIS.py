"""
Simple demonstration of CQED-CIS method on MgH+ diatomic with
a bondlength of 2.2 Angstroms coupled to a photon with energy 4.75 eV.
This photon energy is chosen to be in resonance with the |X> -> |A>
(ground to first singlet excited-state) transition at the bondlength of 2.2 
Angstroms.  Three calculations will be performed:

1. Electric field vector has 0 magnitude (\lambda = (0, 0, 0)) and 0 energy
   allowing direct comparison to ordinary CIS

2. Electric field vector is z-polarized with magnitude 0.0125 atomic units 
   (\lambda = (0, 0, 0.125) and photon has energy 4.75 eV, which will split
   the eigenvectors proportional to the field strength, allowing comparison to 
   results in Figure 3 (top) in [McTague:2021:ChemRxiv] 

3. Electric field vector is z-polarized (\lambda = (0, 0, 0.0125)) and photon has complex
   energy 4.75 - 0.22i eV, allowing comparison to results in Figure 3 (bottom) in [McTague:2021:ChemRxiv]


"""

__authors__ = ["Jon McTague", "Jonathan Foley"]
__credits__ = ["Jon McTague", "Jonathan Foley"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2021-08-19"

# ==> Import Psi4, numpy, and helper_CS_CQED_CIS <==
import psi4
import numpy as np
from helper_CS_CQED_CIS import *
from psi4.driver.procrouting.response.scf_response import tdscf_excitations

# Set Psi4 & NumPy Memory Options
psi4.set_memory("2 GB")
psi4.core.set_output_file("output.dat", False)

numpy_memory = 2


mol_str = """
Mg
H 1 2.2
symmetry c1
1 1
"""

# options dict
options_dict = {
    "basis": "cc-pVDZ",
    "save_jk": True,
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
}

# set psi4 options and geometry
psi4.set_options(options_dict)
mol = psi4.geometry(mol_str)

om_1 = 0
lam_1 = np.array([0.0, 0.0, 0.0])

om_2 = 4.75 / psi4.constants.Hartree_energy_in_eV 
lam_2 = np.array([0.0, 0.0, 0.0125])

om_3 = 4.75 / psi4.constants.Hartree_energy_in_eV - 0.22j
lam_3 = np.array([0.0, 0.0, 0.0125])


n_states = 5

# run psi4 SCF
psi4_rhf_e, wfn = psi4.energy("scf/cc-pVDZ", return_wfn=True, molecule=mol)

# calculate the excited-state energies and save them to a dictionary called 'res'
res = tdscf_excitations(wfn, states=n_states, triplets="NONE", tda=True)

# parse res for excitation energies
psi4_excitation_e = [r["EXCITATION ENERGY"] for r in res]

# run cs_cqed_cis() for the three cases
cqed_cis_1 = cs_cqed_cis(lam_1, om_1, mol_str, options_dict)
cqed_cis_2 = cs_cqed_cis(lam_2, om_2, mol_str, options_dict)
cqed_cis_3 = cs_cqed_cis(lam_3, om_3, mol_str, options_dict)

cqed_cis_e_1 = cqed_cis_1["CQED-CIS ENERGY"]
scf_e_1 = cqed_cis_1["CQED-RHF ENERGY"]

cqed_cis_e_2 = cqed_cis_2["CQED-CIS ENERGY"]
scf_e_2 = cqed_cis_1["CQED-RHF ENERGY"]

cqed_cis_e_3 = cqed_cis_3["CQED-CIS ENERGY"]
scf_e_3 = cqed_cis_3["CQED-RHF ENERGY"]

# collect every other excitation energy for case 1 since cqed-cis will be doubly
# degenerate when omega = 0 and lambda = 0,0,0
cqed_cis_e_1 = cqed_cis_e_1[::2]

print("\n    PRINTING RESULTS FOR CASE 1: NO COUPLING")
print("\n    CASE 1 RHF Energy (Hatree):                    %.8f" % psi4_rhf_e)
print("    CASE 1 CQED-RHF Energy (Hartree):              %.8f" % scf_e_1)
print(
    "    CASE 1 CIS LOWEST EXCITATION ENERGY (eV)        %.4f"
    % (psi4_excitation_e[0] * psi4.constants.Hartree_energy_in_eV)
)
print(
    "    CASE 1 CQED-CIS LOWEST EXCITATION ENERGY (eV)   %.4f"
    % (np.real(cqed_cis_e_1[1]) * psi4.constants.Hartree_energy_in_eV)
)

print(
    "\n    PRINTING RESULTS FOR CASE 2: HBAR * OMEGA = 4.75 eV, LAMBDA = (0, 0, 0.0125) A.U."
)
print(
    "\n    CASE 2 |X,0> -> |LP> Energy (eV)                %.4f"
    % (np.real(cqed_cis_e_2[1]) * psi4.constants.Hartree_energy_in_eV)
)
print(
    "    CASE 2 |X,0> -> |UP> Energy (eV)                %.4f"
    % (np.real(cqed_cis_e_2[2]) * psi4.constants.Hartree_energy_in_eV)
)

print(
    "\n    PRINTING RESULTS FOR CASE 3: HBAR * OMEGA = (4.75 - 0.22i) eV, LAMBDA = (0, 0, 0.0125) A.U."
)
print(
    "\n    CASE 3 |X,0> -> |LP> Energy (eV)                %.4f"
    % (np.real(cqed_cis_e_3[1]) * psi4.constants.Hartree_energy_in_eV)
)
print(
    "    CASE 3 |X,0> -> |UP> Energy (eV)                %.4f\n"
    % (np.real(cqed_cis_e_3[2]) * psi4.constants.Hartree_energy_in_eV)
)

# check to see that the CQED-RHF energy matches ordinary RHF energy for case 1
psi4.compare_values(psi4_rhf_e, scf_e_1, 8, "CASE 1 SCF E")

# check to see if first CQED-CIS excitation energy matches first CIS excitation energy for case 1
psi4.compare_values(cqed_cis_e_1[1], psi4_excitation_e[0], 8, "CASE 1 CQED-CIS E")


# check to see if first CQED-CIS excitation energy matches value from [McTague:2021:ChemRxiv] Figure 3 for case 2
# This still needs to be corrected in the paper!
psi4.compare_values(cqed_cis_e_2[1], 0.1655708380, 8, "CASE 2 CQED-CIS E")
