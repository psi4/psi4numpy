"""
A pure-Py script for computing IR intensities from CCQC reference files.

References:
- Equations from [Yamaguchi:1993] and [CCQC:Project2]
- Conversion factors from NIST
"""

__authors__ = "Dominic A. Sirianni"
__credits__ = ["Dominic A. Sirianni", "Andy C. Simmonet", "Zachary  Glick",
               "C. David Sherrill", "Daniel R. Nascimento"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-11-15"

import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Setup
numpy_memory = 2 

# Unit conversions
hartree2joule = 4.359744650e-18 # E_h -> J
sqang2sqmeter = 1.e20           # A^2 -> m^2
amu2kg = 1.660539040e-27        # amu -> kg
omega2nu = 1./(2*np.pi)         # w -> nu
e2coulomb = 1.602177e-19        # electrons -> Coulombs
m2km = 1.e-3                    # Meters -> Kilometers                 
bohr2angstrom = 0.52917721067   # Bohr -> Angstrom
debye2eAngstrom =  0.208194     # Debye -> e.AA

# Fundamental constants
eps0 = 8.854187817e-12          # Vacuum permittivity
Ke = 1./(4*np.pi*eps0)          # Coulomb force constant
c = 2.997925e8                  # Speed of light
N_A = 6.022140e23               # Avogadro's Number

# Derived conversion factors from above (thanks to Zack Glick!)
au2kmmol = e2coulomb**2 / amu2kg * Ke / c**2 * N_A * m2km
natI2kmmol = au2kmmol * (debye2eAngstrom**2)

# Geometry from ccqc/file11
coords = np.array([[-1.2602962432,       0.0000000000,       0.0000000000],
                   [ 1.2602962432,       0.0000000000,       0.0000000000],
                   [-2.3303912628,      -1.7258024530,       0.0000000000],
                   [ 2.3303912628,      -1.7258024530,       0.0000000000],
                   [-2.3303912628,       1.7258024530,       0.0000000000],
                   [ 2.3303912628,       1.7258024530,       0.0000000000]])

masses = np.array([12., 12., 1.008, 1.008, 1.008, 1.008])

# Convert Hessian from Bohr to Angstrom
Hess = np.loadtxt('ccqc/file15.dat', skiprows=1).reshape(18,18) / (bohr2angstrom ** 2)
dipdir = np.loadtxt('ccqc/file17.dat', skiprows=1).reshape(3,18)

# Mass weight Hessian
M = np.diag(1/np.sqrt(np.repeat(masses, 3))) # Matrix for mass weighting
mH = M.T.dot(Hess).dot(M) # [CCQC:Project2] Eqn. 9

# Mass-weighted Normal modes from Hessian
k2, Lxm = np.linalg.eigh(mH) # [CCQC:Project2] Eqn. 12

freq = k2 * hartree2joule
freq *= (sqang2sqmeter / amu2kg)

# To display IR frequencies in MHz (1/s), uncomment next line
#print(np.sqrt(freq) * omega2nu / 1.e6)

# Un-mass-weight the normal modes
Lx = M.dot(Lxm) # [CCQC:Project2] Eqn. 14

# Transform dipole derivatives to normal coordinates
dmdQ = dipdir.dot(Lx) # [CCQC:Project2] Eqn. 15

# Compute IR intensities according to [CCQC:Project2] Eqn. 16
I = np.zeros(dmdQ.shape[1])

print(" ==> IR Intensities of Ethylene @ HF/sto-3g <== ")
print(" Mode     (D^2 / AA^2 amu) \t a.u. \t\tkm/mol ")
print("-------------------------------------------------------")
for i in range(dmdQ.shape[1]):
    I[i] = dmdQ[:,i].dot(dmdQ[:,i]) # Native units in D^2 / AA^2 amu
    print('  %s   \t\t%6.4f  \t%6.4f  \t%6.4f' % (i, I[i], I[i] * bohr2angstrom**2, natI2kmmol*I[i]))

