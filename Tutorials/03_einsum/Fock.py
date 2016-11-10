import numpy as np
import psi4
import time

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

# Build the fock matrix using loops, while keeping track of time
t_floop_0 = time.time()
Floop = np.zeros((nbf, nbf))
for p in range(nbf):
    for q in range(nbf):
        Floop[p, q] = H[p, q]
        for r in range(nbf):
            for s in range(nbf):
                Floop[p, q] += 2 * I[p, q, r, s] * D[r, s]
                Floop[p, q] -=     I[p, r, q, s] * D[r, s]

t_floop = time.time() - t_floop_0

# Build the fock matrix using einsum, while keeping track of time
t_einsum_0 = time.time()
J = np.einsum('pqrs,rs', I, D) 
K = np.einsum('prqs,rs', I, D)
F = H + 2 * J - K

t_einsum = time.time() - t_einsum_0

# Make sure the correct answer is obtained
print('The loop and einsum fock builds match:    %s\n' % np.allclose(F, Floop))
# Print out relative times for explicit loop vs einsum Fock builds
print('Time for loop Fock build:\t {:2.4f} seconds'.format(t_floop))
print('Time for einsum Fock build:\t {:2.4f} seconds'.format(t_einsum))
print('Fock builds with einsum are {:3.4f} times faster than Python loops!'.format(t_floop / t_einsum))
