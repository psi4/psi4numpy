import numpy as np

molecule mol {
O
H 1 1.1
H 1 1.1 2 105
symmetry c1
}

set {
 basis = sto-3g
}


# Compute integrals and convert the resulting matrices to numpy arrays
mints = MintsHelper()
T = np.array(mints.ao_kinetic())
V = np.array(mints.ao_potential())
nbf = T.shape[0]
I = np.array(mints.ao_eri()).reshape(nbf, nbf, nbf, nbf)
# One-electron hamiltonian is the sum of the kinetic and potential elements
H = T + V

# Build a random density matrix
D = np.arange(nbf*nbf).reshape(nbf, nbf)
D /= np.mean(D)

# Build the fock matrix using loops
Floop = np.zeros((nbf, nbf))
for p in range(nbf):
    for q in range(nbf):
        Floop[p, q] = H[p, q]
        for r in range(nbf):
            for s in range(nbf):
                Floop[p, q] += 2 * I[p, q, r, s] * D[r, s]
                Floop[p, q] -=     I[p, r, q, s] * D[r, s]

# Build the fock matrix using einsum
J = np.einsum('pqrs,rs', I, D) 
K = np.einsum('prqs,rs', I, D)
F = H + 2 * J - K
# Make sure the correct answer is obtained
print 'The loop and einsum fock builds match:    %s' % np.allclose(F, Floop)

# As a bonus lets build the fock matrix in a slightly different way
J = np.einsum('pqrs,rs', I, D) 
# This is equivalent to the above exchange build
K = np.einsum('pqrs,qs', I, D)
Falt = H + 2 * J - K
# Make sure the correct answer is obtained
print 'The fock and alternate fock builds match: %s' % np.allclose(F, Falt)
