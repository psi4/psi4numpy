Symmetry-Adapted-Perturbation-Theory
====================================

It is advantageous to have a SAPT helper object that simplifies many routine SAPT tasks such as computation of the intermolecular interaction operator (\tilde{V}) in the dimer centered basis set.

Indices used:
- a - occupied indices of monomer A
- b - occupied indices of monomer B
- r - virtual indices of monomer A
- s - virtual indices of monomer B

The helper SAPT object can be initialized as follows:

```python
sapt = helper_SAPT(psi4, energy, dimer, memory=8)
```

As the helper is external to the main psithon input we must pass both `psi4` and `energy` global objects from the input file.
The former contains many useful tools such as MintsHelper while the latter is the main energy driver for psi4 (needed for RHF here).
In addition, we set the memory limit to 8GB (default 2GB).

The `sapt` object now contains all of the information that we need to compute the following:
- `v`: Two electron repulsion integrals in the MO basis (e.g. `sapt.v('arar')`)
- `s`: Overlap integrals in the MO basis (e.g. `sapt.s('ab')`)
- `eps`: SCF eigenvalues (e.g. `sapt.eps('a')`)
- `potential`: Electrostatic potential of a monomer (e.g. `sapt.potential('bb', 'A')`)
- `vt`: Intermolecular interaction operator \tilde{V} (e.g. `sapt.vt('arar')`)
- `chf`: Computes CPHF orbitals for each monomer (e.g. `chf('A')`)
 
Where for all 4-index quantities the order of the indices is as follows:

`\tilde{V}_{0, 1}^{2, 3}`

For example, the expression for first order electrostatics is as follows:

`4 * \tilde{V}_{a, b}^{a, b}`

Computation of the \tilde{V} operator can be accomplished as:

`vt_abab = sapt.vt('abab')`

`vt_abab` is a 4-index tensor that needs to be summed over following einsum convention:

`Elst10 = 4 * np.einsum('abab', vt_abab)`

By first building a tool to compute arbitary \tilde{V} tensors we can, in princple, compute up to third order SAPT (if the accompanying T1 and T2 amplitudes are obtain either through psi4 or the psi4numpy CCSD implementation). 

