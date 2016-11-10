## Einsum
Numpy has a built-in arbitrary tensor contraction engine called ```np.einsum```.
Not surprisingly the notation should be given in Einstein summation notation.
The Fock matrix can built by first building the J (Coulomb matrix) and K (Exchange matrix) quantities:

```
J_{pq} = I_{pqrs} * D_{rs}
K_{pq} = I_{prqs} * D_{rs}
F_{pq} = H_{pq} + 2 * J_{pq} - K_{pq}
```

 - H: Two-index one-electron hamiltonian 
 - D: Two-index density matrix
 - I: Four-index electron repulsion integrals
 - F: Two-index Fock matrix

We can build the required integrals in the following way: 
```python
# Compute integrals and convert the resulting matrices to numpy arrays
mints = psi4.core.MintsHelper(basis)
T = np.array(mints.ao_kinetic())
V = np.array(mints.ao_potential())
I = np.array(mints.ao_eri())

# One-electron hamiltonian is the sum of the kinetic and potential elements
H = T + V
```

Using standard loops, building the Fock matrix would look like the following:
```python
F = np.zeros(nbf, nbf)
for p in range(nbf):
    for q in range(nbf):
        F[p, q] += H[p, q]
        for r in range(nbf):
            for s in range(nbf):
                F[p, q] += 2 * I[p, q, r, s] * D[r, s]
                F[p, q] -=     I[p, r, q, s] * D[r, s]
```

The Fock matrix can be built using ```np.einsum``` as follows:
```python
J = np.einsum('pqrs,rs', I, D) 
K = np.einsum('prqs,rs', I, D)
F = H + 2 * J - K
```

Keep in mind that the Fock matrix is one of the simpler quantum chemistry objects and already takes up ~8 lines of code, we can see how beneficial ```np.einsum``` is.
In addition, pure python loops are quite slow for this kind of operation; depending on the number of basis function a pure C++ implementation can be many orders of magnitude faster.
While the python for loops can be very useful to understand exactly what is happening, explicitly writing out the loops is best left to a lower level language implementation.





