import numpy as np
import psi4

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 105
""")

# Construct a Psi4 basis set
basis = psi4.core.BasisSet.build(mol, target="STO-3G")
nbf = basis.nbf()

# Compute integrals and convert the resulting matrices to numpy arrays
mints = psi4.core.MintsHelper(basis)
T = np.array(mints.ao_kinetic())
V = np.array(mints.ao_potential())
I = np.array(mints.ao_eri())

# One-electron hamiltonian is the sum of the kinetic and potential elements
H = T + V

# Build a symmetric random density matrix
D = np.random.rand(nbf, nbf)
D += D.T

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
print('The loop and einsum fock builds match:    %s' % np.allclose(F, Floop))
