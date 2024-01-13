"""
A reference implementation for computing IR intensities.

References:
- Equations from [Yamaguchi:1993]
- Conversion factors from NIST
"""

__authors__ = ["Kirk C. Pearce", "Dominic A. Sirianni"]
__credits__ = ["Kirk C. Pearce" , "Dominic A. Sirianni", 
               "Andy C. Simmonet", "Zachary  Glick",
               "C. David Sherrill", "Daniel R. Nascimento"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2020-11-11"

import numpy as np
import psi4
from psi4 import *
np.set_printoptions(precision=10, linewidth=200, suppress=True)
psi4.core.set_output_file('output.dat', False)

# Setup
numpy_memory = 2 

mol = psi4.geometry("""
C  -1.2602962432       0.0000000000       0.0000000000
C   1.2602962432       0.0000000000       0.0000000000
H  -2.3303912628      -1.7258024530       0.0000000000
H   2.3303912628      -1.7258024530       0.0000000000
H  -2.3303912628       1.7258024530       0.0000000000
H   2.3303912628       1.7258024530       0.0000000000

units bohr
symmetry c1
""")

psi4.set_options({'basis': 'dz',
                  'scf_type': 'direct',
                  'd_convergence': 9,
                  'e_convergence': 11,
                 })

# Run SCF frequency calculation for testing
e, wfn = psi4.frequencies('scf', return_wfn=True)

# Relevant Variables
natoms = mol.natom()

# Get array of atomic masses in a.u.
masses = np.array([mol.mass(i) * psi4.constants.conversion_factor("amu", "atomic_unit_of_mass") for i in range(mol.natom())])

# Get Hessian from wavefunction
Hess = np.asarray(wfn.hessian())
print("\nMolecular Hessian (a.u.):\n", Hess)

# Mass weight Hessian
M = np.diag(1 / np.sqrt(np.repeat(masses, 3))) # Matrix for mass weighting
mH = M.T.dot(Hess).dot(M) 
print("\nMass-weighted Hessian (hartree/(bohr^2 amu)):\n", mH * psi4.constants.conversion_factor("amu", "atomic_unit_of_mass"))

# Mass-weighted normal modes from Hessian
k2, Lxm = np.linalg.eigh(mH) 
print("\nEigenvalues (hartree/(bohr^2 amu)):\n", k2 * psi4.constants.conversion_factor("hartree/(bohr^2 * atomic_unit_of_mass)", "hartree/(bohr^2 * amu)"))
print("\nEigenvalues (J/kg*m^2 = s^-2):\n", k2 * psi4.constants.conversion_factor("hartree/(bohr^2 * atomic_unit_of_mass)", "s^-2"))
print("\nEigenvectors (unitless):\n", Lxm) 

# Calculate normal modes
normal_modes = []
mode = 3 * natoms - 1
print("\nNormal Modes (cm^-1):")
while mode >= 6:
    if k2[mode] >= 0.0:
        normal_modes.append(np.sqrt(k2[mode]))
        print("%.2f" % (np.sqrt(k2[mode]) * psi4.constants.conversion_factor("hartree^(1/2)/(bohr * atomic_unit_of_mass^(1/2))", "cm^-1") / (2 * np.pi)))
    else:
        normal_modes.append(np.sqrt(abs(k2[mode])))
        print("%.2fi" % (np.sqrt(abs(k2[mode])) * psi4.constants.conversion_factor("hartree^(1/2)/(bohr * atomic_unit_of_mass^(1/2))", "cm^-1") / (2 * np.pi)))
    mode -= 1

# Test frequencies
psi_freq = wfn.frequency_analysis['omega'].data[6:]
psi_freq = np.flip(psi_freq, 0)
for i in range(len(normal_modes)):
    psi4.compare_values(psi_freq[i], normal_modes[i] * psi4.constants.conversion_factor("hartree^(1/2)/(bohr * atomic_unit_of_mass^(1/2))", "cm^-1") / (2 * np.pi), 6, "FREQ-TEST")

# Read in dipole derivatives
dipder = wfn.variables().get("CURRENT DIPOLE GRADIENT", None)
dipder = np.asarray(dipder)
print("\nDipole Derivatives (D/A):\n", dipder * psi4.constants.conversion_factor("e * bohr / bohr", "debye / angstrom"))

# Un-mass-weight the eigenvectors (now have units of m_e^(-1/2))
Lx = M.dot(Lxm) 

# Normal Coordinates transformation matrix
S = np.flip(Lx, 1)[:,:len(normal_modes)]
print("\nNormal Coordinates transformation matrix (amu**-(1/2)):\n", S * psi4.constants.conversion_factor("atomic_unit_of_mass^(-1/2)", "amu^(-1/2)"))

# Transform dipole derivatives to normal coordinates
dip_grad = np.einsum('ij,jk->ik', dipder.T, S, optimize=True)
print("\nDipole Gradient in Normal Coordinate Basis ((D/A)/amu^(1/2)):\n", dip_grad * psi4.constants.conversion_factor("(e * bohr)/(bohr * atomic_unit_of_mass^(1/2))", "debye / (angstrom * amu^(1/2))"))

# Conversion factor for (D/A)^2/amu to km/mol:
# (thanks to Zack Glick!)
#
psi_na = psi4.constants.na # Avogdro's number
psi_alpha = psi4.constants.get("fine-structure constant") # finite-structure constant
#
# Conversion factor for taking IR intensities from ((D/A)^2 / amu) to a.u. ((e a0 / a0)^2 / me):
conv_kmmol = psi4.constants.conversion_factor("debye^2 / (angstrom^2 * amu)", "(e^2 * bohr^2)/(bohr^2 * atomic_unit_of_mass)")
# Multiply by (1 / (4 pi eps_0)) in a.u. (1 / (e^2 / a0 Eh)) to get to units of (Eh a0 / me)
conv_kmmol *= psi4.constants.conversion_factor("(e^2 * bohr^2)/(bohr^2 * atomic_unit_of_mass) * (1 / (e^2 / (bohr * hartree)))", "hartree * bohr / atomic_unit_of_mass")
# Multiply by (Na pi / 3 c^2) in a.u. (Na = mol^-1; c = Eh / alpha^2 me) to get to units of a0/mol
conv_kmmol *= psi_na * np.pi * (1 / 3) * psi_alpha**2
# Convert to units of km/mol
conv_kmmol *= psi4.constants.conversion_factor("bohr / mol", "km / mol")
#
conv_ir_DAamu2kmmol = conv_kmmol # (D/A)^2/amu -> km/mol

# Compute IR Intensities
IR_ints = np.zeros(len(normal_modes))
for i in range(3):
    for j in range(len(normal_modes)):
        IR_ints[j] += dip_grad[i][j] * dip_grad[i][j]

print("\n\nVibrational Frequencies and IR Intensities:\n------------------------------------------\n")
print("  mode        frequency               IR intensity\n========================================================")
print("           cm-1      hartrees      km/mol  (D/A)**2/amu \n--------------------------------------------------------")
for i in range(len(normal_modes)):
    print("  %3d    %7.2f     %8.6f     %7.3f      %6.4f" % (i + 1, normal_modes[i] * psi4.constants.conversion_factor("hartree^(1/2)/(bohr * atomic_unit_of_mass^(1/2))", "cm^-1") / (2 * np.pi), normal_modes[i] * psi4.constants.conversion_factor("hartree^(1/2)/(bohr * atomic_unit_of_mass^(1/2))", "hartree") / (2 * np.pi), IR_ints[i] * psi4.constants.conversion_factor("(e^2 * bohr^2)/(bohr^2 * atomic_unit_of_mass)", "debye^2 / (angstrom^2 * amu)") * conv_ir_DAamu2kmmol, IR_ints[i] * psi4.constants.conversion_factor("(e^2 * bohr^2)/(bohr^2 * atomic_unit_of_mass)", "debye^2 / (angstrom^2 * amu)")))

# Test IR intensities
psi_irints = wfn.frequency_analysis["IR_intensity"].data[6:]
psi_irints = np.flip(psi_irints, 0)
for i in range(len(normal_modes)):
    psi4.compare_values(psi_irints[i], IR_ints[i] * psi4.constants.conversion_factor("(e^2 * bohr^2)/(bohr^2 * atomic_unit_of_mass)", "debye^2 / (angstrom^2 * amu)") * conv_ir_DAamu2kmmol, 6, "IR-INTS-TEST")

