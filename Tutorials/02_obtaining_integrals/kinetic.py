import numpy as np

molecule mol {
O
H 1 1.1
H 1 1.1 2 105
}

set {
 basis = sto-3g
}

mints = MintsHelper()

T = mints.ao_kinetic()

print T

np_T = np.array(T)

# Print the numpy array
print np_T
