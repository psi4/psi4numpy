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

# Useful Constants
psi_c = psi4.constants.get("Natural unit of velocity") * 100                                                    # speed of light in cm/s
psi_na = psi4.constants.na                                                                                      # Avogdro's number
psi_alpha = qcel.constants.get("fine-structure constant")                                                       # finite-structure constant

# Unit Conversions
#
# Energy conversions
hartree2joule = psi4.constants.get("Atomic unit of energy")                                                     # Eh -> J
#
# Distance conversions
psi_bohr2m = psi4.constants.bohr2m                                                                              # bohr -> m
sqbohr2sqmeter = psi_bohr2m ** 2                                                                                # bohr^2 -> m^2
psi_bohr2angstroms = psi4.constants.bohr2angstroms                                                              # bohr -> Ang
#
# Mass conversions
amu2kg = psi4.constants.get("Atomic mass constant")                                                             # amu -> kg
psi_au2amu = psi4.constants.au2amu                                                                              # m_e -> amu
#
# Dipole moment type conversions
psi_dipmom_au2debye = psi4.constants.dipmom_au2debye                                                            # e a0 -> D (C m)
conv_dip_grad_au2DAamu = (psi_dipmom_au2debye / psi_bohr2angstroms) * (1 / np.sqrt(psi_au2amu))                 # (e a0 / a0)/(m_e^(1/2)) -> (D/A)/(amu^(1/2))
conv_cgs = 6.46047502185 * (10**(-36))                                                                          # (e a0)^2 -> (esu * cm)^2
#
# IR frequency conversions
omega2nu = 1./(psi_c*2*np.pi)                                                                                   # omega (s^-1)  -> nu (cm^-1)
psi4_hartree2wavenumbers = psi4.constants.hartree2wavenumbers                                                   # Eh -> cm^-1
conv_ir_au2DAamu = conv_dip_grad_au2DAamu ** 2                                                                  # (e a0 / a0)^2/m_e -> (D/A)^2/amu
conv_freq_au2wavenumbers = np.sqrt((1 / psi_au2amu) * hartree2joule / (sqbohr2sqmeter * amu2kg)) * omega2nu     # (Eh / (bohr^2 m_e)) -> cm^-1
conv_freq_wavenumbers2hartree = 1 / psi4_hartree2wavenumbers                                                    # cm^-1 -> Eh

# Convert IR intensities from (D/A)^2/amu to km/mol:
# (thanks to Zack Glick!)
#
# Conversion factor for taking IR intensities from ((D/A)^2 / amu) to a.u. ((e a0 / a0)^2 / me):
conv_kmmol = (1 / conv_dip_grad_au2DAamu) ** 2
# Multiply by (1 / (4 pi eps_0)) in a.u. (1 / (e^2 / a0 Eh)) to get to units of (Eh a0 / me)
# Multiply by (Na pi / 3 c^2) in a.u. (Na = mol^-1; c = Eh / alpha^2 me) to get to units of a0/mol
conv_kmmol *= psi_na * np.pi * (1 / 3) * psi_alpha**2
# Convert to units of km/mol
conv_kmmol *= psi_bohr2m * (1 / 1000)
#
conv_ir_DAamu2kmmol = conv_kmmol                                                                                # (D/A)^2/amu -> km/mol



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
masses = np.array([mol.mass(i) * (1 / psi_au2amu) for i in range(mol.natom())])

# Get Hessian from wavefunction
Hess = np.asarray(wfn.hessian())
print("\nMolecular Hessian (a.u.):\n", Hess)

# Mass weight Hessian
M = np.diag(1 / np.sqrt(np.repeat(masses, 3))) # Matrix for mass weighting
mH = M.T.dot(Hess).dot(M) 
#print("\nMass-weighted Hessian (a.u.):\n", mH)
#print("\nMass-weighted Hessian (hartree/(bohr^2 amu)):\n", mH * (1 / psi_au2amu))

# Mass-weighted normal modes from Hessian
k2, Lxm = np.linalg.eigh(mH) 
#print("\nEigenvalues (a.u.):\n", k2) 
#print("\nEigenvectors (unitless):\n", Lxm) 
print("\nEigenvalues (hartree/(bohr^2 amu)):\n", k2 * (1 / psi_au2amu)) 
#print("\nEigenvalues (J/kg*m^2 = s^-2):\n", k2 * (1 / psi_au2amu) * hartree2joule / (sqbohr2sqmeter * amu2kg)) 

normal_modes = []
mode = 3 * natoms - 1
print("\nNormal Modes (cm^-1):")
while mode >= 6:
    if k2[mode] >= 0.0:
        normal_modes.append(np.sqrt(k2[mode]))
        print("%.2f" % (np.sqrt(k2[mode]) * conv_freq_au2wavenumbers))
    else:
        normal_modes.append(np.sqrt(abs(k2[mode])))
        print("%.2fi" % (np.sqrt(abs(k2[mode])) * conv_freq_au2wavenumbers))
    mode -= 1

# Test frequencies
psi_freq = np.flip(wfn.frequencies().to_array(), 0)
for i in range(len(normal_modes)):
    psi4.compare_values(psi_freq[i], normal_modes[i] * conv_freq_au2wavenumbers, 6, "FREQ-TEST")

# Read in dipole derivatives
dipder = wfn.variables().get("CURRENT DIPOLE GRADIENT", None)
dipder = np.asarray(dipder)
#print("\nDipole Derivatives (a.u.):\n", dipder)
print("\nDipole Derivatives (D/A):\n", dipder * (psi_dipmom_au2debye / psi_bohr2angstroms))

# Un-mass-weight the eigenvectors (now have units of m_e^(-1/2))
Lx = M.dot(Lxm) 

# Normal Coordinates transformation matrix
S = np.flip(Lx, 1)[:,:len(normal_modes)]
#print("\nNormal Coordinates transformation matrix (a.u.):\n", S)
#print("\nNormal Coordinates transformation matrix (amu**-(1/2)):\n", S * (1 / np.sqrt(psi_au2amu)))

# Transform dipole derivatives to normal coordinates
dip_grad = np.einsum('ij,jk->ik', dipder.T, S, optimize=True)
print("\nDipole Gradient in Normal Coordinate Basis (D/(A*amu**(1/2))):\n", dip_grad * conv_dip_grad_au2DAamu)

# Compute IR Intensities
IR_ints = np.zeros(len(normal_modes))
for i in range(3):
    for j in range(len(normal_modes)):
        IR_ints[j] += dip_grad[i][j] * dip_grad[i][j]
#print("\nIR Intensities (a.u.):\n", IR_ints)
#print("\nIR Intensities ((D/A)^2/amu):\n", IR_ints * conv_ir_au2DAamu)
#print("\nIR Intensities km/mol:\n", IR_ints * conv_ir_au2DAamu * conv_ir_DAamu2kmmol)

print("\n\nVibrational Frequencies and IR Intensities:\n------------------------------------------\n")
print("  mode        frequency               IR intensity\n========================================================")
print("           cm-1      hartrees      km/mol  (D/A)**2/amu \n--------------------------------------------------------")
for i in range(len(normal_modes)):
    print("  %3d    %7.2f     %8.6f     %7.3f      %6.4f" % (i + 1, normal_modes[i] * conv_freq_au2wavenumbers, normal_modes[i] * conv_freq_au2wavenumbers * conv_freq_wavenumbers2hartree, IR_ints[i] * conv_ir_au2DAamu * conv_ir_DAamu2kmmol, IR_ints[i] * conv_ir_au2DAamu))

