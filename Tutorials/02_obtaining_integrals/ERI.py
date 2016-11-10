import numpy as np
import psi4

mol = psi4.geometry("""
H
H 1 0.74
""")

# Construct a Psi4 basis set
basis = psi4.core.BasisSet.build(mol, target="sto-3g")

# Build a mints object
mints = psi4.core.MintsHelper(basis)

I = np.array(mints.ao_eri())
# Note that the ERI comes out as a 4D array
print("The shape of I is %s\n" % str(I.shape))

print("I:")
print(I)
