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
# Any one-body integral will do!
T = np.array(mints.ao_kinetic())
nbf = T.shape[0]

# Public
I = np.array(mints.ao_eri()).reshape(nbf, nbf, nbf, nbf)
print I.shape

# Beta
I = np.array(mints.ao_eri())
print I.shape

