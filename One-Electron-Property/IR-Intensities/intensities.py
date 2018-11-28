"""
A reference implementation for computing IR intensities.

References:
- Equations from [Yamaguchi:1993]
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
psi4.core.set_output_file('output.dat', False)

# Setup
numpy_memory = 2 

# Unit conversions
hartree2joule = psi4.constants.get("Atomic unit of energy")
sqang2sqmeter = 1.e20           # A^2 -> m^2
amu2kg = psi4.constants.get("Atomic mass constant")
omega2nu = 1./(2*np.pi)         # w -> nu
e2coulomb = psi4.constants.get("Atomic unit of charge")
m2km = 1.e-3                    # Meters -> Kilometers                 
bohr2angstrom = psi4.constants.get("Atomic unit of length")
debye2eAngstrom =  0.208194     # Debye -> e.AA

# Fundamental constants
eps0 = psi4.constants.get("Electric constant")
Ke = 1./(4*np.pi*eps0)          # Coulomb force constant
c = psi4.constants.get("Natural unit of velocity")
N_A = psi4.constants.get("Avogadro constant")

# Derived conversion factors from above (thanks to Zack Glick!)
au2kmmol = e2coulomb**2 / amu2kg * Ke / c**2 * N_A * m2km

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

masses = np.array([mol.mass(i) for i in range(mol.natom())])

# Get Hessian from wavefunction
e, wfn = psi4.frequencies('scf', return_wfn=True)
Hess = np.asarray(wfn.hessian())

# Read in dipole derivatives
dipdir = wfn.dipole_gradient().np.T

# Mass weight Hessian
M = np.diag(1/np.sqrt(np.repeat(masses, 3))) # Matrix for mass weighting
mH = M.T.dot(Hess).dot(M) # [CCQC:Project2] Eqn. 9

# Mass-weighted Normal modes from Hessian
k2, Lxm = np.linalg.eigh(mH) # [CCQC:Project2] Eqn. 12
freq = k2 * hartree2joule
freq *= (sqang2sqmeter / amu2kg)

# Un-mass-weight the normal modes
Lx = M.dot(Lxm) # [CCQC:Project2] Eqn. 14

# Transform dipole derivatives to normal coordinates
dmdQ = dipdir.dot(Lx) # [CCQC:Project2] Eqn. 15

# Compute IR intensities according to [CCQC:Project2] Eqn. 16
I = np.zeros(dmdQ.shape[1])

print(" ==> IR Intensities of Ethylene @ HF/DZ <== ")
print(" Mode     a.u.  km/mol ")
print("-----------------------")
for i in range(dmdQ.shape[1]):
    I[i] = dmdQ[:,i].dot(dmdQ[:,i]) # Native units in D^2 / AA^2 amu
    print('  %s   \t%6.4f \t%6.4f' % (i, I[i], au2kmmol*I[i]))

