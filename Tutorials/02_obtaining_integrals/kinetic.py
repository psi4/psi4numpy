import numpy as np
import psi4

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 105
""")

# Construct a Psi4 basis set
basis = psi4.core.BasisSet.build(mol, target="STO-3G")

# Build a mints object
mints = psi4.core.MintsHelper(basis)

# AO Kinetic integrals
T = mints.ao_kinetic()
print T

# Print the numpy array
np_T = np.array(T)
print np_T
